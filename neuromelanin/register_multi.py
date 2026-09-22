"""
Мульти-T1 регистрация: перебор стратегий, оценка, выбор лучшей.

Стратегии:
  1. standard_<plane>:      NM --Rigid--> T1_<plane> --SyN--> MNI
  2. intermediate_<other>_via_<anchor>:
                            NM --Rigid--> T1_<other> --Rigid--> T1_<anchor>
                            --SyN--> MNI
  3. direct_Affine:         NM --Affine--> MNI
  4. direct_Rigid:          NM --Rigid--> MNI

Оценка: SN χ mean в QSM после автокалибровки сдвига.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np


def log(msg, level="info"):
    prefix = {"info": "  ", "step": "\n▶", "ok": "  ✅", "err": "  ❌",
              "warn": "  ⚠️ "}.get(level, "  ")
    print(f"{prefix} {msg}", flush=True)


def _load_nii(path: Path) -> np.ndarray:
    img = nib.load(str(path), mmap=False)
    img = nib.as_closest_canonical(img)
    return img.get_fdata(dtype=np.float32)


def find_t1_planes(data_dir: Path) -> dict:
    """Ищет T1_*.nii* в data_dir. Возвращает {'sag': Path, 'tra': Path, ...}."""
    planes = {}
    for p in sorted(data_dir.glob("T1*.nii*")):
        name = p.name.lower()
        if "sag" in name:
            planes.setdefault("sag", p)
        elif "tra" in name:
            planes.setdefault("tra", p)
        elif "cor" in name:
            planes.setdefault("cor", p)
        else:
            planes.setdefault(p.stem, p)
    return planes


def _t1_fov_mm(path: Path) -> float:
    """Наибольший FOV (мм) по трём осям — прокси полноты объёма."""
    try:
        img = nib.load(str(path), mmap=False)
        shape = img.shape
        zooms = img.header.get_zooms()[:3]
        fovs = [s * z for s, z in zip(shape, zooms)]
        return float(max(fovs))
    except Exception:
        return 0.0


# =============================================================================
# Скоринг
# =============================================================================

def score_atlas(atlas_path: Path, qsm_path: Path,
                strategy_name: str, verbose: bool = True) -> dict:
    """Оценивает качество атласа в пространстве QSM."""
    result = {"strategy": strategy_name, "score": -1e9}

    try:
        atlas = _load_nii(atlas_path)
        qsm = _load_nii(qsm_path)
    except Exception as e:
        result["error"] = f"load failed: {e}"
        return result

    if atlas.shape != qsm.shape:
        result["error"] = f"shape mismatch: {atlas.shape} vs {qsm.shape}"
        return result

    sn_full = (atlas == 2)
    crus_full = (atlas == 1)
    n_sn = int(sn_full.sum())
    n_crus = int(crus_full.sum())
    result["n_sn"] = n_sn
    result["n_crus"] = n_crus

    if n_sn < 100 or n_crus < 100:
        result["error"] = f"too few voxels: SN={n_sn}, Crus={n_crus}"
        return result

    coords = np.argwhere(sn_full)
    cx = float(coords[:, 0].mean())
    idx = np.indices(sn_full.shape)[0]
    sn_l_orig = sn_full & (idx < cx)
    sn_r_orig = sn_full & (idx >= cx)

    z_frac = float(coords[:, 2].mean() / atlas.shape[2])
    result["z_frac"] = z_frac

    best_chi = -1e9
    best_shift = 0
    best_chi_l = 0.0
    best_chi_r = 0.0
    for shift in range(0, 31, 2):
        if shift == 0:
            sl, sr = sn_l_orig, sn_r_orig
        else:
            sl = np.zeros_like(sn_l_orig)
            sl[:, :, :-shift] = sn_l_orig[:, :, shift:]
            sr = np.zeros_like(sn_r_orig)
            sr[:, :, shift:] = sn_r_orig[:, :, :-shift]

        if sl.sum() < 50 or sr.sum() < 50:
            continue

        chi_l = float(qsm[sl].mean())
        chi_r = float(qsm[sr].mean())
        chi_mean = 0.5 * (chi_l + chi_r)

        if chi_mean > best_chi:
            best_chi = chi_mean
            best_shift = shift
            best_chi_l = chi_l
            best_chi_r = chi_r

    if best_chi < -1e8:
        result["error"] = "no valid shift found"
        return result

    denom = max(abs(best_chi_l), abs(best_chi_r))
    lr_ratio = min(abs(best_chi_l), abs(best_chi_r)) / denom if denom > 1e-6 else 0.0

    z_penalty = abs(z_frac - 0.5)
    score = (
        100.0 * best_chi
        + 0.3 * lr_ratio
        - 2.0 * z_penalty
        - 0.001 * best_shift
    )

    result.update({
        "score": score,
        "best_shift": best_shift,
        "chi_mean": best_chi,
        "chi_l": best_chi_l,
        "chi_r": best_chi_r,
        "lr_ratio": lr_ratio,
    })

    if verbose:
        log(f"  [{strategy_name}]  score={score:+.4f}  "
            f"χ_mean={best_chi:+.4f}  χ_L={best_chi_l:+.4f}  "
            f"χ_R={best_chi_r:+.4f}")
        log(f"                   L/R={lr_ratio:.2f}  "
            f"z_frac={z_frac:.2f}  shift={best_shift}  "
            f"n_SN={n_sn}  n_Crus={n_crus}")

    return result


# =============================================================================
# Запуск одной стратегии
# =============================================================================

def run_register(strategy: dict, args, work_subdir: Path) -> Path | None:
    work_subdir.mkdir(parents=True, exist_ok=True)
    atlas_out = work_subdir / "atlas_native.nii.gz"
    meta_out = work_subdir / "reg_meta.json"

    cmd = [
        sys.executable, str(args.register_script),
        "--nm-mri", str(args.nm_mri),
        "--mni-t1", str(args.mni_t1),
        "--atlas-mni", str(args.atlas_mni),
        "--output", str(atlas_out),
        "--meta", str(meta_out),
        "--type", "SyN",
        "--threads", str(args.threads),
    ]

    mode = strategy["mode"]
    if mode == "standard":
        cmd.extend(["--subject-t1", str(strategy["t1"])])
    elif mode == "intermediate":
        cmd.extend([
            "--subject-t1", str(strategy["t1_anchor"]),
            "--t1-anchor", str(strategy["t1_anchor"]),
            "--t1-other", str(strategy["t1_other"]),
        ])
    elif mode == "direct":
        cmd.extend([
            "--subject-t1", "none",
            "--direct-nm-to-mni",
            "--direct-method", strategy["method"],
        ])
    else:
        log(f"  ⚠ Неизвестный режим: {mode}", "warn")
        return None

    if args.verbose_cmd:
        log(f"  $ {' '.join(str(c) for c in cmd)}")

    try:
        proc = subprocess.run(cmd, check=False, capture_output=True,
                              text=True, timeout=args.timeout_per_strategy)
    except subprocess.TimeoutExpired:
        log(f"  ❌ Таймаут {args.timeout_per_strategy}s", "warn")
        return None

    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        log(f"  ❌ Упал (code {proc.returncode}): {tail}", "warn")
        return None

    if not atlas_out.exists():
        log(f"  ❌ atlas_native не создан", "warn")
        return None

    return atlas_out


# =============================================================================
# Список стратегий
# =============================================================================

def build_strategies(t1_planes: dict) -> list[dict]:
    strategies = []

    # --- Стандартные ---
    for plane_name, t1_path in sorted(t1_planes.items()):
        strategies.append({
            "name": f"standard_{plane_name}",
            "mode": "standard",
            "t1": t1_path,
        })

    # --- Intermediate: anchor = T1 с наибольшим FOV ---
    if len(t1_planes) >= 2:
        anchor_name = max(t1_planes.keys(),
                          key=lambda k: _t1_fov_mm(t1_planes[k]))
        anchor_path = t1_planes[anchor_name]
        anchor_fov = _t1_fov_mm(anchor_path)

        log(f"  Anchor для intermediate: {anchor_name} "
            f"(FOV={anchor_fov:.0f} мм)")

        for plane_name, t1_path in sorted(t1_planes.items()):
            if plane_name == anchor_name:
                continue
            strategies.append({
                "name": f"intermediate_{plane_name}_via_{anchor_name}",
                "mode": "intermediate",
                "t1_anchor": anchor_path,
                "t1_other": t1_path,
            })

    # --- Direct ---
    strategies.append({"name": "direct_Affine", "mode": "direct",
                        "method": "Affine"})
    strategies.append({"name": "direct_Rigid", "mode": "direct",
                        "method": "Rigid"})

    return strategies


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Мульти-T1 регистрация: перебор стратегий, выбор лучшей."
    )
    p.add_argument("--nm-mri", type=Path, required=True)
    p.add_argument("--qsm", type=Path, required=True)
    p.add_argument("--mni-t1", type=Path, required=True)
    p.add_argument("--atlas-mni", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--register-script", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--timeout-per-strategy", type=int, default=600)
    p.add_argument("--strategies", type=str, default=None)
    p.add_argument("--verbose-cmd", action="store_true")
    p.add_argument("--keep-all", action="store_true")
    args = p.parse_args()

    t1_planes = find_t1_planes(args.data_dir)
    if not t1_planes:
        log(f"❌ T1_*.nii* не найдены в {args.data_dir}", "err")
        sys.exit(1)

    log(f"Найдены T1-плоскости:", "step")
    for k, v in t1_planes.items():
        log(f"  {k:6s}  {v.name}")

    all_strategies = build_strategies(t1_planes)
    if args.strategies:
        wanted = {s.strip() for s in args.strategies.split(",")}
        all_strategies = [s for s in all_strategies if s["name"] in wanted]
        log(f"Фильтр стратегий: {sorted(wanted)}")

    log(f"\nСтратегий к прогону: {len(all_strategies)}", "step")
    for s in all_strategies:
        log(f"  • {s['name']}")

    work_root = args.output.parent / "_reg" / "multi_strategies"
    work_root.mkdir(parents=True, exist_ok=True)

    results = []
    t_start = time.time()

    for i, strat in enumerate(all_strategies, 1):
        log(f"\n{'=' * 65}")
        log(f"  [{i}/{len(all_strategies)}] {strat['name']}")
        log(f"{'=' * 65}")

        subdir = work_root / strat["name"]
        if subdir.exists():
            shutil.rmtree(subdir, ignore_errors=True)
        subdir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        atlas_path = run_register(strat, args, subdir)
        elapsed_reg = time.time() - t0

        if atlas_path is None:
            results.append({
                "strategy": strat["name"],
                "score": -1e9,
                "error": "registration failed",
                "elapsed_s": elapsed_reg,
            })
            log(f"  ⏱ {elapsed_reg:.0f}s — пропуск")
            continue

        log(f"  ✅ Атлас за {elapsed_reg:.0f}s")
        metrics = score_atlas(atlas_path, args.qsm, strat["name"])
        metrics["elapsed_s"] = elapsed_reg
        metrics["atlas_path"] = str(atlas_path)
        results.append(metrics)

    log(f"\n{'=' * 65}")
    log(f"  СВОДКА (total {time.time() - t_start:.0f}s)")
    log(f"{'=' * 65}")
    log(f"  {'Стратегия':<32} {'score':>10} {'χ_mean':>10} "
        f"{'L/R':>6} {'z_frac':>7} {'shift':>6}")
    log(f"  {'-' * 80}")

    valid = [r for r in results if "error" not in r and r["score"] > -1e8]
    for r in sorted(results, key=lambda x: -x.get("score", -1e9)):
        if "error" in r:
            log(f"  {r['strategy']:<32}  ERROR: {r['error']}")
        else:
            log(f"  {r['strategy']:<32} {r['score']:>+10.4f} "
                f"{r['chi_mean']:>+10.4f} "
                f"{r['lr_ratio']:>6.2f} "
                f"{r['z_frac']:>7.2f} "
                f"{r['best_shift']:>6d}")

    if not valid:
        log(f"\n❌ Все стратегии провалились", "err")
        sys.exit(1)

    best = max(valid, key=lambda x: x["score"])
    log(f"\n🏆 Лучшая: {best['strategy']}  "
        f"(score={best['score']:+.4f}, χ={best['chi_mean']:+.4f})", "ok")

    if best["score"] < -0.05:
        log(f"\n⚠ ВНИМАНИЕ: score < -0.05 — атлас, вероятно, вне SN.", "warn")
        log(f"   Проверьте debug_*/atlas_native.nii.gz визуально.", "warn")
        log(f"   Возможно, требуется ручная сегментация.", "warn")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["atlas_path"], args.output)
    log(f"  → {args.output}")

    meta = {
        "strategy": "multi_t1",
        "best_strategy": best["strategy"],
        "best_score": best["score"],
        "best_chi_mean": best["chi_mean"],
        "best_shift": best["best_shift"],
        "n_strategies_tried": len(all_strategies),
        "n_strategies_valid": len(valid),
        "elapsed_total_s": time.time() - t_start,
        "all_results": results,
    }
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(json.dumps(meta, indent=2, default=str))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "best": best, "all": results,
        }, indent=2, default=str))

    if not args.keep_all:
        log(f"\nОчистка промежуточных результатов...")
        for subdir in work_root.iterdir():
            if subdir.is_dir():
                shutil.rmtree(subdir, ignore_errors=True)


if __name__ == "__main__":
    main()