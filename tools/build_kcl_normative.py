"""
Прогон base + robust NM-CNR по всем KCL-HC и сбор статистики.

Обновляет normative_db.json ключом per_sequence.tse_3t_kcl.
При n < 20 SD не записывается (get_normative подставит floor).

Использование:
    python tools/build_kcl_normative.py \
        --kcl-dir /path/to/KCL \
        --out /path/to/QSM/neuromelanin/normative_db.json \
        [--subjects sub-001,sub-002,...] \
        [--no-robust] \
        [--csv /path/to/per_subject.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MIN_N_FOR_TRUSTED_SD = 20


# =============================================================================
# Утилиты
# =============================================================================

def find_subjects(kcl_dir: Path) -> list[str]:
    nifti_root = kcl_dir / "nifti"
    if not nifti_root.exists():
        return []
    return sorted([
        d.name for d in nifti_root.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])


def extract_masks(atlas_path: Path, out_dir: Path) -> dict | None:
    if not atlas_path.exists():
        return None
    try:
        atlas_img = nib.load(str(atlas_path))
        data = np.asarray(atlas_img.dataobj).astype(int)
    except Exception:
        return None

    crus = (data == 1)
    sn = (data == 2)
    if crus.sum() < 50 or sn.sum() < 50:
        return None

    coords = np.argwhere(sn)
    cx = float(coords[:, 0].mean())
    idx = np.indices(sn.shape)[0]
    sn_l = (sn & (idx < cx)).astype(np.uint8)
    sn_r = (sn & (idx >= cx)).astype(np.uint8)

    if sn_l.sum() < 5 or sn_r.sum() < 5:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name, arr):
        p = out_dir / f"{name}.nii.gz"
        nib.save(
            nib.Nifti1Image(arr.astype(np.uint8),
                            atlas_img.affine, atlas_img.header),
            str(p),
        )
        return p

    return {
        "CrusCerebri": save("CrusCerebri", crus.astype(np.uint8)),
        "SN_VTA_L":    save("SN_VTA_L",   sn_l),
        "SN_VTA_R":    save("SN_VTA_R",   sn_r),
    }


def run_base(nm_path: Path, masks: dict, tmp_out: Path) -> float | None:
    try:
        from neuromelanin.analyze import compute_nm_cnr
        masks_dict = {"paths": {
            "CrusCerebri": masks["CrusCerebri"],
            "SN_VTA_L":    masks["SN_VTA_L"],
            "SN_VTA_R":    masks["SN_VTA_R"],
        }}
        res = compute_nm_cnr(nm_path, masks_dict, tmp_out)
        if not res:
            return None
        return float(res["cnr_mean"])
    except Exception as e:
        print(f"    base упал: {e}")
        return None


def run_robust(nm_path: Path, masks: dict) -> float | None:
    try:
        from neuromelanin.cnr_robust import compute_nm_cnr_robust
        res = compute_nm_cnr_robust(
            nm_mri_path=nm_path,
            sn_l_path=masks["SN_VTA_L"],
            sn_r_path=masks["SN_VTA_R"],
            crus_path=masks["CrusCerebri"],
            verbose=False,
        )
        return float(res["cnr_mean"])
    except Exception as e:
        print(f"    robust упал: {e}")
        return None


def read_kcl_xlsx(xlsx: Path) -> dict:
    try:
        import pandas as pd
        df = pd.read_excel(xlsx)
        return {
            str(r["Subject ID"]): float(r["Mean"])
            for _, r in df.iterrows()
        }
    except Exception:
        return {}


# =============================================================================
# Обновление normative_db.json
# =============================================================================

def update_normative_db(json_path: Path,
                        key: str,
                        mean: float,
                        sd: float | None,
                        n: int,
                        source: str) -> None:
    """
    sd=None — SD не записывается (или записывается null),
    get_normative подставит floor.
    """
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            data = {}
    else:
        data = {}

    data.setdefault("per_sequence", {})

    entry = {
        "mean": round(float(mean), 6),
        "n":    int(n),
        "source": source,
        "updated": datetime.now().isoformat(),
    }
    if sd is not None:
        entry["sd"] = round(float(sd), 6)
    else:
        entry["sd"] = None

    data["per_sequence"][key] = entry
    data["updated_at"] = datetime.now().isoformat()

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    sd_note = f"sd={entry['sd']}" if sd is not None else "sd=null (floor)"
    print(f"\n  → {json_path} обновлён: ключ={key}, "
          f"mean={entry['mean']}, {sd_note}, n={n}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kcl-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Путь к normative_db.json")
    p.add_argument("--subjects", type=str, default=None,
                   help="Список через запятую; по умолчанию все sub-*")
    p.add_argument("--no-robust", action="store_true")
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()

    kcl = args.kcl_dir.expanduser().resolve()
    if not (kcl / "nifti").exists():
        print(f"❌ Нет {kcl}/nifti/")
        sys.exit(1)

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]
    else:
        subjects = find_subjects(kcl)

    print(f"\n{'='*65}")
    print(f"  BATCH KCL-NORMATIVE")
    print(f"{'='*65}")
    print(f"  KCL:      {kcl}")
    print(f"  Subjects: {len(subjects)}")
    print(f"  Robust:   {not args.no_robust}")
    print(f"{'='*65}\n")

    kcl_means = read_kcl_xlsx(kcl / "results" / "NM_CNR_SNVTA_Results.xlsx")
    if kcl_means:
        print(f"  Найдено в KCL xlsx: {len(kcl_means)} субъектов\n")

    rows = []
    base_vals, robust_vals = [], []

    for i, sub in enumerate(subjects, 1):
        print(f"[{i}/{len(subjects)}] {sub}")

        nm_path = kcl / f"nifti/{sub}/anat/{sub}_NM.nii.gz"
        atlas = kcl / f"output/{sub}/anat/{sub}_midbrain_atlas_space-NM.nii.gz"

        if not nm_path.exists():
            print(f"  ⚠ NM-MRI не найден: {nm_path.name}")
            continue
        if not atlas.exists():
            print(f"  ⚠ Атлас не найден: {atlas.name}")
            continue

        with tempfile.TemporaryDirectory(prefix=f"kcl_{sub}_") as tmp:
            tmp_dir = Path(tmp)
            masks = extract_masks(atlas, tmp_dir)
            if masks is None:
                print(f"  ⚠ Маски не извлечены")
                continue

            tmp_out = tmp_dir / "out"
            tmp_out.mkdir(parents=True, exist_ok=True)

            base = run_base(nm_path, masks, tmp_out)
            rob = None if args.no_robust else run_robust(nm_path, masks)

        b_s = f"{base:.6f}" if base is not None else "—"
        r_s = f"{rob:.6f}"  if rob  is not None else "—"
        k_s = (f"{kcl_means[sub]:.6f}"
               if sub in kcl_means else "—")
        print(f"  base={b_s}  robust={r_s}  KCL={k_s}")

        if base is not None: base_vals.append(base)
        if rob  is not None: robust_vals.append(rob)

        rows.append({
            "subject": sub,
            "base": base, "robust": rob,
            "kcl_xlsx": kcl_means.get(sub),
        })

    # --- Статистика ---
    print(f"\n{'='*65}")
    print(f"  СТАТИСТИКА")
    print(f"{'='*65}")

    def stats(vals, name):
        if not vals:
            print(f"  {name}: нет данных")
            return None, None, 0
        a = np.asarray(vals, dtype=np.float64)
        m = float(np.mean(a))
        s = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
        print(f"  {name:8s}  n={len(a):3d}  "
              f"mean={m:.6f} ({m*100:.2f}%)  "
              f"sd={s:.6f} ({s*100:.2f}%)")
        return m, s, len(a)

    m_b, sd_b, n_b = stats(base_vals,   "base")
    m_r, sd_r, n_r = stats(robust_vals, "robust")

    if kcl_means:
        kcl_vals = [kcl_means[r["subject"]] for r in rows
                    if r["subject"] in kcl_means]
        stats(kcl_vals, "KCL xlsx")

    if m_b is not None and m_r is not None:
        d = abs(m_b - m_r)
        rel = 100 * d / m_b if m_b else 0
        print(f"\n  Δ(base vs robust) = {d:.6f} ({rel:.2f}%)")

    # --- CSV ---
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w") as f:
            f.write("subject,base,robust,kcl_xlsx\n")
            for r in rows:
                f.write(f"{r['subject']},"
                        f"{r['base'] or ''},"
                        f"{r['robust'] or ''},"
                        f"{r['kcl_xlsx'] or ''}\n")
        print(f"\n  CSV: {args.csv}")

    # --- Update JSON ---
    if m_b is not None and n_b > 0:
        source = (f"KCL Neuromelanin-MRI, Crus-mode, TSE 3T, "
                  f"n={n_b}, built {datetime.now().date()}")

        sd_to_write = sd_b if n_b >= MIN_N_FOR_TRUSTED_SD else None
        if sd_to_write is None:
            print(f"\n  ⚠ n={n_b} < {MIN_N_FOR_TRUSTED_SD} — "
                  f"SD не записываю (get_normative подставит floor)")

        update_normative_db(
            args.out,
            key="tse_3t_kcl",
            mean=m_b, sd=sd_to_write, n=n_b,
            source=source,
        )
    else:
        print("\n  ⚠ base-значений нет, JSON не обновлён")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)