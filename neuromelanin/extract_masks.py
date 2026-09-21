"""
ЭТАП 1.5: Извлечение масок из атласа (SimpleITK, изолированный процесс).

Опции:
  --erode-sn N         радиус эрозии SN
  --sn-shift N         фиксированный сдвиг SN
  --auto-sn-shift      автокалибровка сдвига по QSM
  --sn-shift-range     диапазон для автокалибровки
"""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def log(msg, level="info"):
    prefix = {"info": "  ", "step": "\n▶", "ok": "  ✅", "err": "  ❌",
              "warn": "  ⚠️ "}.get(level, "  ")
    print(f"{prefix} {msg}", flush=True)


def calibrate_sn_shift(sn_left, sn_right, qsm_data,
                        shift_range=range(0, 31, 2),
                        min_voxels=200):
    """Автоподбор латерального сдвига SN по максимуму χ mean."""
    log(f"Автокалибровка сдвига (диапазон {shift_range.start}.."
        f"{shift_range.stop - 1}, шаг {shift_range.step})...")

    results = []
    for shift in shift_range:
        if shift == 0:
            sn_l_s, sn_r_s = sn_left, sn_right
        else:
            sn_l_s = np.zeros_like(sn_left)
            sn_l_s[:, :, :-shift] = sn_left[:, :, shift:]
            sn_r_s = np.zeros_like(sn_right)
            sn_r_s[:, :, shift:] = sn_right[:, :, :-shift]

        n_l = int(sn_l_s.sum())
        n_r = int(sn_r_s.sum())
        if n_l < min_voxels or n_r < min_voxels:
            continue

        vals_l = qsm_data[sn_l_s > 0]
        vals_r = qsm_data[sn_r_s > 0]
        chi_l = float(np.mean(vals_l))
        chi_r = float(np.mean(vals_r))
        chi_mean = 0.5 * (chi_l + chi_r)
        p95_l = float(np.percentile(vals_l, 95))
        p95_r = float(np.percentile(vals_r, 95))
        p95_mean = 0.5 * (p95_l + p95_r)

        results.append({
            "shift": shift, "n_l": n_l, "n_r": n_r,
            "chi_l": chi_l, "chi_r": chi_r, "chi_mean": chi_mean,
            "p95_l": p95_l, "p95_r": p95_r, "p95_mean": p95_mean,
        })

    if not results:
        log("  ⚠ Нет валидных сдвигов", "warn")
        return 0, []

    log(f"  {'shift':>6} {'χ SN-L':>10} {'χ SN-R':>10} {'χ mean':>10} "
        f"{'p95 mean':>10} {'n_L':>6} {'n_R':>6}")
    for r in results:
        log(f"  {r['shift']:>6} {r['chi_l']:>+10.4f} {r['chi_r']:>+10.4f} "
            f"{r['chi_mean']:>+10.4f} {r['p95_mean']:>+10.4f} "
            f"{r['n_l']:>6} {r['n_r']:>6}")

    best = max(results, key=lambda r: r["chi_mean"])
    log(f"  ✅ Оптимальный сдвиг: {best['shift']} вокселей "
        f"(χ mean = {best['chi_mean']:+.4f} ppm)", "ok")
    return best["shift"], results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas-native", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--erode-sn", type=int, default=3)
    p.add_argument("--sn-shift", type=int, default=0)
    p.add_argument("--auto-sn-shift", action="store_true")
    p.add_argument("--sn-shift-range", type=str, default="0,30,2")
    args = p.parse_args()

    log("Извлечение масок (SimpleITK, изолированный процесс)", "step")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log("Читаю атлас...")
    atlas = sitk.ReadImage(str(args.atlas_native))
    log(f"  Атлас: size={atlas.GetSize()}, "
        f"spacing={tuple(f'{s:.3f}' for s in atlas.GetSpacing())}")

    log("Читаю reference (QSM)...")
    ref = sitk.ReadImage(str(args.reference))
    log(f"  QSM:   size={ref.GetSize()}, "
        f"spacing={tuple(f'{s:.3f}' for s in ref.GetSpacing())}")

    log("Ресемплинг атласа в пространство QSM...")
    atlas_in_ref = sitk.Resample(
        atlas, ref, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
    )
    data = sitk.GetArrayFromImage(atlas_in_ref)

    n_crus = int((data == 1).sum())
    n_sn = int((data == 2).sum())
    log(f"  Crus (label 1): {n_crus} вокселей")
    log(f"  SN   (label 2): {n_sn} вокселей")

    if n_crus < 100 or n_sn < 100:
        raise RuntimeError(
            f"Атлас не содержит меток: Crus={n_crus}, SN={n_sn}"
        )

    # =========================================================================
    # ПРОВЕРКА FLIP ПО Z
    # =========================================================================
    coords_sn = np.argwhere(data == 2)
    center_z = float(coords_sn[:, 0].mean())  # SimpleITK: (Z, Y, X)
    total_z = data.shape[0]
    z_frac = center_z / total_z

    log(f"  Центр SN по Z: {center_z:.1f} из {total_z} ({100 * z_frac:.0f}%)")

    if z_frac < 0.25 or z_frac > 0.75:
        log(f"  ⚠ Маска SN {'в нижних' if z_frac < 0.25 else 'в верхних'} "
            f"25% — flip по Z!", "warn")
        log(f"  Применяю flip по оси Z...")

        # Flip по оси Z (в SimpleITK data имеет форму ZYX)
        data = np.flip(data, axis=0).copy()

        # Пересчёт
        coords_sn = np.argwhere(data == 2)
        center_z_new = float(coords_sn[:, 0].mean())
        z_frac_new = center_z_new / total_z
        log(f"  После flip: центр SN по Z = {center_z_new:.1f} "
            f"({100 * z_frac_new:.0f}%)", "ok")

        if z_frac_new < 0.25 or z_frac_new > 0.75:
            log(f"  ⚠ Flip не помог — маска всё ещё не в середине", "warn")

    # Эрозия SN
    log(f"Эрозия SN (радиус {args.erode_sn})...")
    if args.erode_sn > 0:
        sn_img = sitk.GetImageFromArray((data == 2).astype(np.uint8))
        sn_img.CopyInformation(ref)
        radius_vec = [args.erode_sn] * sn_img.GetDimension()
        sn_eroded_img = sitk.BinaryErode(sn_img, radius_vec)
        sn_eroded = sitk.GetArrayFromImage(sn_eroded_img).astype(np.uint8)
    else:
        sn_eroded = (data == 2).astype(np.uint8)

    n_sn_eroded = int(sn_eroded.sum())
    log(f"  SN: {n_sn} → {n_sn_eroded} ({100*n_sn_eroded/max(n_sn, 1):.1f}%)")

    if n_sn_eroded < 100:
        log("  ⚠ Слишком мал — откат эрозии", "warn")
        sn_eroded = (data == 2).astype(np.uint8)
        n_sn_eroded = int(sn_eroded.sum())

    # Разделение L/R
    coords = np.argwhere(sn_eroded > 0)
    center_x = coords[:, 2].mean()
    sn_left = np.zeros_like(sn_eroded)
    sn_right = np.zeros_like(sn_eroded)
    for (z, y, x) in coords:
        if x < center_x:
            sn_left[z, y, x] = 1
        else:
            sn_right[z, y, x] = 1

    log(f"  SN-L: {int(sn_left.sum())}, SN-R: {int(sn_right.sum())}")

    # Сдвиг
    if args.auto_sn_shift:
        try:
            start, stop, step = (int(x) for x in args.sn_shift_range.split(","))
            shift_range = range(start, stop, step)
        except ValueError:
            raise ValueError(f"Неверный --sn-shift-range: {args.sn_shift_range}")

        qsm_data = sitk.GetArrayFromImage(ref).astype(np.float32)
        best_shift, table = calibrate_sn_shift(
            sn_left, sn_right, qsm_data, shift_range=shift_range,
        )
        calib_path = args.out_dir / "sn_shift_calibration.json"
        calib_path.write_text(json.dumps({
            "auto_calibrated": True,
            "best_shift": best_shift,
            "shift_range": f"{start},{stop},{step}",
            "results": table,
        }, indent=2))
        log(f"  Калибровка: {calib_path.name}", "ok")
        effective_shift = best_shift
    else:
        effective_shift = args.sn_shift
        log(f"Фиксированный сдвиг: {effective_shift}")

    # Применение сдвига
    if effective_shift != 0:
        s = abs(effective_shift)
        sn_left_qsm = np.zeros_like(sn_left)
        sn_left_qsm[:, :, :-s] = sn_left[:, :, s:]
        sn_right_qsm = np.zeros_like(sn_right)
        sn_right_qsm[:, :, s:] = sn_right[:, :, :-s]
    else:
        sn_left_qsm = sn_left
        sn_right_qsm = sn_right

    sn_qsm = ((sn_left_qsm > 0) | (sn_right_qsm > 0)).astype(np.uint8)

    # Сохранение
    log("Сохранение масок...")
    crus_mask = (data == 1).astype(np.uint8)

    def save_mask(arr_zyx, name):
        img = sitk.GetImageFromArray(arr_zyx.astype(np.uint8))
        img.CopyInformation(ref)
        path = args.out_dir / f"{name}.nii.gz"
        sitk.WriteImage(img, str(path), useCompression=True)
        log(f"  → {path.name}  ({int(arr_zyx.sum())} вокс.)")

    save_mask(crus_mask, "CrusCerebri_native")
    save_mask(sn_eroded, "SN_VTA_native")
    save_mask(sn_left, "SN_VTA_L_native")
    save_mask(sn_right, "SN_VTA_R_native")

    if effective_shift != 0:
        save_mask(sn_qsm, "SN_VTA_qsm")
        save_mask(sn_left_qsm, "SN_VTA_L_qsm")
        save_mask(sn_right_qsm, "SN_VTA_R_qsm")

    combined = np.zeros_like(crus_mask)
    combined[crus_mask > 0] = 3
    combined[sn_left > 0] = 1
    combined[sn_right > 0] = 2
    save_mask(combined, "all_masks_combined")

    log(f"✅ Готово. Сдвиг SN = {effective_shift}", "ok")


if __name__ == "__main__":
    main()