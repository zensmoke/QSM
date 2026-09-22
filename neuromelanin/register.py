"""
ЭТАП 1: ANTs регистрация атласа KCL в пространство NM-MRI.

Три режима:
  1) Стандартный:      NM --Rigid--> T1_x --SyN--> MNI
  2) Через intermediate: NM --Rigid--> T1_a --Rigid--> T1_anchor --SyN--> MNI
  3) Прямой NM→MNI:    NM --Affine/Rigid/SyN--> MNI (без T1)
"""
import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import ants
import numpy as np


def log(msg, level="info"):
    prefix = {"info": "  ", "step": "\n▶", "ok": "  ✅", "err": "  ❌",
              "warn": "  ⚠️ "}.get(level, "  ")
    print(f"{prefix} {msg}", flush=True)


def safe_path(src: Path, tmp: Path) -> Path:
    """Копирует файл в tmp (без пробелов/не-ASCII в пути)."""
    src = Path(src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Нет файла: {src}")
    dst = tmp / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


# =============================================================================
# РЕЖИМ 1: ПРЯМАЯ РЕГИСТРАЦИЯ NM → MNI
# =============================================================================

def run_direct(nm_img, mni_img, atlas_img, args, t0):
    log(f"\n[DIRECT] Регистрация NM → MNI "
        f"({args.direct_method}), БЕЗ T1")

    reg_nm_mni = ants.registration(
        fixed=mni_img, moving=nm_img,
        type_of_transform=args.direct_method, verbose=False,
    )

    atlas_native = ants.apply_transforms(
        fixed=nm_img, moving=atlas_img,
        transformlist=reg_nm_mni["invtransforms"],
        interpolator="multiLabel",
    )
    n = int((atlas_native.numpy() > 0).sum())
    log(f"  атлас в NM: {n} вокселей")

    arr = atlas_native.numpy()
    if (arr == 2).sum() > 0:
        coords = arr.nonzero()
        cz = coords[2].mean()
        total_z = arr.shape[2]
        z_frac = cz / total_z
        log(f"  SN: {int((arr == 2).sum())} вокс., "
            f"центр z = {cz:.1f} / {total_z} ({100 * z_frac:.0f}%)")
        if z_frac < 0.25 or z_frac > 0.75:
            log(f"  ⚠ SN вне середины", "warn")

    if n < 500:
        log(f"  ⚠ Мало вокселей (< 500)", "warn")

    debug_dir = args.output.parent / "debug_direct"
    debug_dir.mkdir(parents=True, exist_ok=True)

    nm_in_mni = ants.apply_transforms(
        fixed=mni_img, moving=nm_img,
        transformlist=reg_nm_mni["fwdtransforms"],
    )
    ants.image_write(nm_in_mni,     str(debug_dir / "NM_in_MNI.nii.gz"))
    ants.image_write(atlas_native,  str(debug_dir / "atlas_native.nii.gz"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(atlas_native, str(args.output))

    meta = {
        "type": f"{args.direct_method}_direct_nm_to_mni",
        "type_t1_mni": "none",
        "type_nm_t1": "none",
        "elapsed_s": time.time() - t0,
        "timestamp": datetime.now().isoformat(),
        "antspyx_version": ants.__version__,
        "n_nonzero_atlas_nm": n,
    }
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(json.dumps(meta, indent=2))

    log(f"\n✅ Атлас: {args.output}")
    log(f"⏱ Время: {time.time() - t0:.0f}s")


# =============================================================================
# РЕЖИМ 2: ЧЕРЕЗ ПРОМЕЖУТОЧНЫЙ T1 (intermediate)
# =============================================================================

def run_intermediate(nm_img, t1_other_img, t1_anchor_img, mni_img,
                     atlas_img, args, t0):
    log(f"\n[INTERMEDIATE] NM → T1_other → T1_anchor → MNI")

    log("  [A] NM → T1_other (Rigid)")
    reg_nm_other = ants.registration(
        fixed=t1_other_img, moving=nm_img,
        type_of_transform="Rigid", verbose=False,
    )

    log("  [B] T1_other → T1_anchor (Rigid)")
    reg_other_anchor = ants.registration(
        fixed=t1_anchor_img, moving=t1_other_img,
        type_of_transform="Rigid", verbose=False,
    )

    log("  [C] T1_anchor → MNI (SyN)")
    reg_anchor_mni = ants.registration(
        fixed=mni_img, moving=t1_anchor_img,
        type_of_transform="SyN", verbose=False,
    )

    log("  [D] Композиция: atlas_MNI → T1_anchor → T1_other → NM")
    atlas_in_other = ants.apply_transforms(
        fixed=t1_other_img, moving=atlas_img,
        transformlist=(
            list(reg_anchor_mni["invtransforms"]) +
            list(reg_other_anchor["invtransforms"])
        ),
        interpolator="multiLabel",
    )
    n_mid = int((atlas_in_other.numpy() > 0).sum())
    log(f"      после T1_other: {n_mid} вокселей")

    atlas_native = ants.apply_transforms(
        fixed=nm_img, moving=atlas_in_other,
        transformlist=reg_nm_other["invtransforms"],
        interpolator="multiLabel",
    )
    n = int((atlas_native.numpy() > 0).sum())
    log(f"      после NM: {n} вокселей")

    if n < 500:
        raise RuntimeError(
            f"Через intermediate провалилось: {n} вокселей"
        )

    arr = atlas_native.numpy()
    if (arr == 2).sum() > 0:
        coords = arr.nonzero()
        cz = coords[2].mean()
        total_z = arr.shape[2]
        log(f"  SN: {int((arr == 2).sum())} вокс., "
            f"центр z = {cz:.1f} / {total_z} "
            f"({100 * cz / total_z:.0f}%)")

    debug_dir = args.output.parent / "debug_intermediate"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ants.image_write(atlas_in_other, str(debug_dir / "atlas_in_T1other.nii.gz"))
    ants.image_write(atlas_native,   str(debug_dir / "atlas_native.nii.gz"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(atlas_native, str(args.output))

    meta = {
        "type": "Rigid + Rigid + SyN (via intermediate)",
        "type_t1_mni": "SyN",
        "type_nm_t1": "Rigid_via_intermediate",
        "n_nonzero_atlas_other": n_mid,
        "n_nonzero_atlas_nm": n,
        "elapsed_s": time.time() - t0,
        "timestamp": datetime.now().isoformat(),
        "antspyx_version": ants.__version__,
    }
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(json.dumps(meta, indent=2))

    log(f"\n✅ Атлас: {args.output}")
    log(f"⏱ Время: {time.time() - t0:.0f}s")


# =============================================================================
# РЕЖИМ 3: СТАНДАРТНЫЙ
# =============================================================================

def run_standard(nm_img, t1_img, mni_img, atlas_img, args, t0):
    log(f"\n[STANDARD] NM → T1 → MNI")

    log(f"\n[1/3] Регистрация T1 → MNI ({args.type})...")

    MIN_ATLAS_VOXELS = 5000

    def do_reg(reg_type):
        log(f"  Пробую {reg_type}...")
        return ants.registration(
            fixed=mni_img, moving=t1_img,
            type_of_transform=reg_type, verbose=False,
        )

    def count_atlas_in_t1(reg):
        atlas_t = ants.apply_transforms(
            fixed=t1_img, moving=atlas_img,
            transformlist=reg["invtransforms"],
            interpolator="multiLabel",
        )
        arr = atlas_t.numpy()
        n1 = int((arr == 1).sum())
        n2 = int((arr == 2).sum())
        return n1 + n2, n1, n2

    reg_t1_mni = do_reg(args.type)
    n_total, n_crus, n_sn = count_atlas_in_t1(reg_t1_mni)
    log(f"    После {args.type}: {n_total} вокселей "
        f"(Crus={n_crus}, SN={n_sn})")

    used_type = args.type

    if n_total < MIN_ATLAS_VOXELS and args.type != "SyN":
        log(f"  ⚠ {args.type} дал {n_total} вокселей "
            f"(< {MIN_ATLAS_VOXELS}) — fallback на SyN", "warn")
        reg_t1_mni = do_reg("SyN")
        n_total, n_crus, n_sn = count_atlas_in_t1(reg_t1_mni)
        log(f"    После SyN: {n_total} вокселей "
            f"(Crus={n_crus}, SN={n_sn})")
        used_type = "SyN"

    if n_total < MIN_ATLAS_VOXELS:
        raise RuntimeError(
            f"Регистрация T1 → MNI провалилась: {n_total} вокселей. "
            f"Попробуйте --t1-other или --direct-nm-to-mni."
        )

    log(f"  ✅ T1 → MNI ({used_type})")

    log("\n[2/3] Регистрация NM → T1 (Rigid)...")
    reg_nm_t1 = ants.registration(
        fixed=t1_img, moving=nm_img,
        type_of_transform="Rigid", verbose=False,
    )
    log(f"  ✅ NM → T1")

    log("\n[3/3] Перенос атласа MNI → NM (пошагово)...")

    log("  [3.1] Атлас MNI → T1...")
    atlas_in_t1 = ants.apply_transforms(
        fixed=t1_img, moving=atlas_img,
        transformlist=reg_t1_mni["invtransforms"],
        interpolator="multiLabel",
    )
    n1 = int((atlas_in_t1.numpy() > 0).sum())
    log(f"       ненулевых: {n1}")
    if n1 < 500:
        raise RuntimeError(f"3.1 провалился: {n1} вокселей")

    log("  [3.2] Атлас T1 → NM...")
    atlas_native = ants.apply_transforms(
        fixed=nm_img, moving=atlas_in_t1,
        transformlist=reg_nm_t1["invtransforms"],
        interpolator="multiLabel",
    )
    n2 = int((atlas_native.numpy() > 0).sum())
    log(f"       ненулевых: {n2}")
    if n2 < 500:
        raise RuntimeError(f"3.2 провалился: {n2} вокселей")

    log(f"  ✅ Атлас в пространстве NM: {n2} вокселей")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(atlas_native, str(args.output))

    meta = {
        "type_t1_mni": used_type,
        "type_nm_t1": "Rigid",
        "elapsed_s": time.time() - t0,
        "timestamp": datetime.now().isoformat(),
        "antspyx_version": ants.__version__,
        "n_nonzero_atlas_nm": n2,
    }
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(json.dumps(meta, indent=2))

    log(f"\n✅ Атлас: {args.output}")
    log(f"⏱ Время: {time.time() - t0:.0f}s")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject-t1", type=Path, required=True,
                   help="T1 (полный FOV). Для direct — 'none'.")
    p.add_argument("--nm-mri", type=Path, required=True)
    p.add_argument("--mni-t1", type=Path, required=True)
    p.add_argument("--atlas-mni", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--type", default="SyN")
    p.add_argument("--threads", type=int, default=4)

    # Прямая регистрация
    p.add_argument("--direct-nm-to-mni", action="store_true")
    p.add_argument("--direct-method", default="Affine",
                   choices=["Affine", "Rigid", "SyN"])

    # Intermediate через вторую T1-плоскость
    p.add_argument("--t1-other", type=Path, default=None,
                   help="Вторая T1-плоскость (не anchor). Intermediate: "
                        "NM → T1_other → T1_anchor → MNI.")
    p.add_argument("--t1-anchor", type=Path, default=None,
                   help="Anchor T1 (обычно sag с наибольшим FOV). "
                        "Если не указан — берётся --subject-t1.")

    args = p.parse_args()

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
    log(f"ANTs {ants.__version__}, threads={args.threads}", "step")

    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="ants_reg_"))
    log(f"tmp: {tmp}", "step")

    try:
        log("Копирую файлы в /tmp...")
        nm_safe = safe_path(args.nm_mri, tmp)
        mni_safe = safe_path(args.mni_t1, tmp)
        atlas_safe = safe_path(args.atlas_mni, tmp)

        log("Загружаю через ANTs...")
        nm_img = ants.image_read(str(nm_safe))
        mni_img = ants.image_read(str(mni_safe))
        atlas_img = ants.image_read(str(atlas_safe))

        log(f"  NM:    shape={nm_img.shape}, spacing="
            f"{tuple(f'{s:.2f}' for s in nm_img.spacing)}")
        log(f"  MNI:   shape={mni_img.shape}")

        # ========== ВЫБОР РЕЖИМА ==========

        # --- Direct ---
        if args.direct_nm_to_mni:
            run_direct(nm_img, mni_img, atlas_img, args, t0)
            return

        # --- Intermediate ---
        if args.t1_other is not None:
            if args.t1_anchor is None:
                anchor_path = args.subject_t1
            else:
                anchor_path = args.t1_anchor

            other_safe = safe_path(args.t1_other, tmp)
            anchor_safe = safe_path(anchor_path, tmp)

            other_img = ants.image_read(str(other_safe))
            anchor_img = ants.image_read(str(anchor_safe))

            log(f"  T1_other:  shape={other_img.shape}, spacing="
                f"{tuple(f'{s:.2f}' for s in other_img.spacing)}")
            log(f"  T1_anchor: shape={anchor_img.shape}, spacing="
                f"{tuple(f'{s:.2f}' for s in anchor_img.spacing)}")

            run_intermediate(nm_img, other_img, anchor_img,
                              mni_img, atlas_img, args, t0)
            return

        # --- Standard ---
        if args.subject_t1 is None or str(args.subject_t1) == "none":
            raise RuntimeError(
                "Стандартный режим требует --subject-t1."
            )

        t1_safe = safe_path(args.subject_t1, tmp)
        t1_img = ants.image_read(str(t1_safe))
        log(f"  T1:    shape={t1_img.shape}, spacing="
            f"{tuple(f'{s:.2f}' for s in t1_img.spacing)}")

        run_standard(nm_img, t1_img, mni_img, atlas_img, args, t0)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()