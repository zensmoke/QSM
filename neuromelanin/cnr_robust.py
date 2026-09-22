"""
Второй метод оценки NM-CNR: slice-wise + ipsilateral + trimmed mean.

Отличия от analyze.compute_nm_cnr:
  1. Референс (Crus) считается ПО-СРЕЗНО, а не по всему объёму.
  2. Ипсилатеральный референс: SN-L → Crus-L, SN-R → Crus-R.
  3. Усечённое среднее (trimmed mean) вместо KDE mode.

Формула:
    CNR_L = median_z[ (mean(NM[SN_L,z]) - trimmean(NM[Crus_L,z])) / trimmean(...) ]
    CNR_R = median_z[ ... ]
    CNR   = (CNR_L + CNR_R) / 2

Совместим по выходу с analyze.compute_nm_cnr (ключ cnr_mean — ДОЛЯ).
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(str(path), mmap=False)
    img = nib.as_closest_canonical(img)
    return (img.get_fdata(dtype=np.float32), img.affine, img.header)


def trimmed_mean(values: np.ndarray, trim: float = 0.1) -> float:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    if v.size < 10:
        return float(v.mean())
    k = int(np.floor(v.size * trim))
    if k == 0:
        return float(v.mean())
    v_sorted = np.sort(v)
    return float(v_sorted[k:-k].mean())


def split_lr(mask: np.ndarray, x_axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    mask = mask.astype(bool)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return mask.copy(), mask.copy()
    center_x = float(coords[:, x_axis].mean())
    idx = np.indices(mask.shape)[x_axis]
    l_mask = mask & (idx < center_x)
    r_mask = mask & (idx >= center_x)
    return l_mask, r_mask


def _check_shapes(nm, sn_l, sn_r, crus) -> None:
    if nm.shape == sn_l.shape == sn_r.shape == crus.shape:
        return
    hints = [
        f"NM-MRI: {nm.shape}",
        f"SN-L:   {sn_l.shape}",
        f"SN-R:   {sn_r.shape}",
        f"Crus:   {crus.shape}",
        "",
    ]
    if nm.shape[:2] != sn_l.shape[:2]:
        hints.append("⚠ Разное in-plane разрешение (X, Y).")
        hints.append("  Скорее всего, маски и NM-MRI из РАЗНЫХ субъектов.")
    if abs(nm.shape[2] - sn_l.shape[2]) > 5:
        hints.append(f"⚠ Разное число срезов по Z: "
                     f"NM={nm.shape[2]}, маски={sn_l.shape[2]}.")
    raise ValueError("Формы не совпадают:\n  " + "\n  ".join(hints))


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def compute_nm_cnr_robust(
    nm_mri_path: Path,
    sn_l_path: Path,
    sn_r_path: Path,
    crus_path: Path,
    trim: float = 0.1,
    min_sn_voxels: int = 5,
    min_crus_voxels: int = 20,
    verbose: bool = True,
) -> dict:
    nm, _, _ = _load_canonical(nm_mri_path)
    sn_l, _, _ = _load_canonical(sn_l_path)
    sn_r, _, _ = _load_canonical(sn_r_path)
    crus, _, _ = _load_canonical(crus_path)

    _check_shapes(nm, sn_l, sn_r, crus)

    sn_l = sn_l > 0.5
    sn_r = sn_r > 0.5
    crus = crus > 0.5

    crus_l, crus_r = split_lr(crus)

    if verbose:
        print(f"[robust] NM shape: {nm.shape}")
        print(f"[robust] SN-L: {int(sn_l.sum())}  SN-R: {int(sn_r.sum())}")
        print(f"[robust] Crus-L: {int(crus_l.sum())}  Crus-R: {int(crus_r.sum())}")

    z_axis = 2

    def per_slice_cnr(sn_mask: np.ndarray, crus_mask: np.ndarray
                      ) -> tuple[list, list, list, list]:
        """
        Возвращает (cnr_list, z_list, ref_list, n_sn_list).
        """
        cnr_slices, z_list, ref_list, n_sn_list = [], [], [], []
        sn_any_z = np.where(sn_mask.any(axis=(0, 1)))[0]
        for z in sn_any_z:
            sn_s = sn_mask[:, :, z]
            cr_s = crus_mask[:, :, z]
            n_sn = int(sn_s.sum())
            n_cr = int(cr_s.sum())
            if n_sn < min_sn_voxels or n_cr < min_crus_voxels:
                continue
            ref = trimmed_mean(nm[:, :, z][cr_s], trim=trim)
            if ref <= 0:
                continue
            sn_val = float(nm[:, :, z][sn_s].mean())
            cnr = (sn_val - ref) / ref

            if verbose:
                print(f"[robust]   z={z:3d}  ref={ref:8.2f}  "
                      f"sn={sn_val:8.2f}  "
                      f"cnr={cnr*100:+7.2f}%  "
                      f"n_sn={n_sn:4d}  n_cr={n_cr:4d}")

            cnr_slices.append(cnr)
            z_list.append(int(z))
            ref_list.append(ref)
            n_sn_list.append(n_sn)

        return cnr_slices, z_list, ref_list, n_sn_list

    cnr_l_slices, z_l, ref_l, n_sn_l = per_slice_cnr(sn_l, crus_l)
    cnr_r_slices, z_r, ref_r, n_sn_r = per_slice_cnr(sn_r, crus_r)

    if verbose:
        print(f"[robust] Срезов с SN-L: {len(z_l)}")
        print(f"[robust] Срезов с SN-R: {len(z_r)}")

    if not cnr_l_slices or not cnr_r_slices:
        raise RuntimeError(
            f"Недостаточно срезов: L={len(cnr_l_slices)}, "
            f"R={len(cnr_r_slices)}. Уменьшите min_*_voxels."
        )

    cnr_l_arr = np.array(cnr_l_slices, dtype=np.float64)
    cnr_r_arr = np.array(cnr_r_slices, dtype=np.float64)

    cnr_l = float(np.median(cnr_l_arr))
    cnr_r = float(np.median(cnr_r_arr))
    cnr_mean = 0.5 * (cnr_l + cnr_r)

    if verbose:
        print(f"[robust] CNR-L = {cnr_l*100:+.2f}%  "
              f"(median из {len(cnr_l_arr)} срезов)")
        print(f"[robust] CNR-R = {cnr_r*100:+.2f}%  "
              f"(median из {len(cnr_r_arr)} срезов)")
        print(f"[robust] CNR mean = {cnr_mean*100:+.2f}%")

    sn_signal = float(np.mean(nm[sn_l | sn_r]))
    crus_signal = float(np.mean(nm[crus]))

    return {
        "nm_mri_file": nm_mri_path.name,
        "signal_sn_mean": sn_signal,
        "signal_crus_mean": crus_signal,
        "signal_ratio": sn_signal / crus_signal if crus_signal else 0.0,

        "cnr_mean": cnr_mean,
        "cnr_left": cnr_l,
        "cnr_right": cnr_r,
        "cnr_mean_pct": cnr_mean * 100,
        "cnr_left_pct": cnr_l * 100,
        "cnr_right_pct": cnr_r * 100,

        "method": "robust_slicewise_ipsilateral",
        "trim": trim,
        "n_slices_left": len(cnr_l_arr),
        "n_slices_right": len(cnr_r_arr),
        "z_slices_left": z_l,
        "z_slices_right": z_r,
        "cnr_left_slices": cnr_l_arr.tolist(),
        "cnr_right_slices": cnr_r_arr.tolist(),
        "ref_left_slices": ref_l,
        "ref_right_slices": ref_r,
        "n_sn_left_slices": n_sn_l,
        "n_sn_right_slices": n_sn_r,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(
        description="Второй метод NM-CNR (slice-wise + ipsilateral + robust)."
    )
    p.add_argument("--nm-mri", type=Path, required=True)
    p.add_argument("--sn-l", type=Path, required=True)
    p.add_argument("--sn-r", type=Path, required=True)
    p.add_argument("--crus", type=Path, required=True)
    p.add_argument("--trim", type=float, default=0.1)
    p.add_argument("--min-sn", type=int, default=5)
    p.add_argument("--min-crus", type=int, default=20)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    result = compute_nm_cnr_robust(
        nm_mri_path=args.nm_mri,
        sn_l_path=args.sn_l,
        sn_r_path=args.sn_r,
        crus_path=args.crus,
        trim=args.trim,
        min_sn_voxels=args.min_sn,
        min_crus_voxels=args.min_crus,
    )

    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\n[robust] Сохранено: {args.output}")