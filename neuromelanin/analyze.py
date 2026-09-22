"""
ЭТАП 2: NM-CNR + QSM + демография (nibabel + scipy, без SimpleITK).
Поддерживает TSE / MTC-GRE / GRE.
"""
import argparse
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import nibabel as nib
import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from neuromelanin.demographics import (
        get_patient_info, compare_with_normative, PatientInfo,
        detect_sequence_type,
    )
    HAS_DEMOGRAPHICS = True
except ImportError as e:
    print(f"[analyze] ⚠ demographics.py недоступен: {e}")
    HAS_DEMOGRAPHICS = False


def log(msg, level="info"):
    prefix = {"info": "  ", "step": "\n▶", "ok": "  ✅", "warn": "  ⚠️ ",
              "err": "  ❌"}.get(level, "  ")
    print(f"{prefix} {msg}", flush=True)


def load_canonical(path: Path) -> tuple:
    img = nib.load(str(path), mmap=False)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine, img.header, img


@dataclass
class MaskStats:
    name: str
    voxels: int
    volume_mm3: float
    centroid: tuple
    bbox: tuple


def extract_masks(masks_dir: Path, out_dir: Path) -> dict:
    log("Извлечение масок (чтение готовых)", "step")

    paths = {
        "CrusCerebri": masks_dir / "CrusCerebri_native.nii.gz",
        "SN_VTA":      masks_dir / "SN_VTA_native.nii.gz",
        "SN_VTA_L":    masks_dir / "SN_VTA_L_native.nii.gz",
        "SN_VTA_R":    masks_dir / "SN_VTA_R_native.nii.gz",
        "combined":    masks_dir / "all_masks_combined.nii.gz",
    }

    missing = [k for k, v in paths.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(f"Маски не найдены: {missing}")

    def get_stats(path, name):
        data, _, header, _ = load_canonical(path)
        mask = data > 0.5
        zooms = tuple(float(z) for z in header.get_zooms()[:3])
        vv = float(np.prod(zooms))
        vox = int(mask.sum())
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return MaskStats(name, 0, 0.0, (0.0, 0.0, 0.0), (0, 0, 0))
        centroid = tuple(float(x) for x in coords.mean(axis=0))
        bbox = tuple(int(x) for x in (coords.max(axis=0) - coords.min(axis=0)))
        return MaskStats(name, vox, vox * vv, centroid, bbox)

    crus_stats = get_stats(paths["CrusCerebri"], "Crus Cerebri")
    sn_stats   = get_stats(paths["SN_VTA"], "SN-VTA")
    l_stats    = get_stats(paths["SN_VTA_L"], "SN-VTA L")
    r_stats    = get_stats(paths["SN_VTA_R"], "SN-VTA R")

    log(f"  Crus Cerebri: {crus_stats.voxels} вокс.")
    log(f"  SN-VTA:       {sn_stats.voxels} вокс.")
    log(f"  SN-VTA L/R:   {l_stats.voxels} / {r_stats.voxels}")
    log("  ✅ Маски прочитаны", "ok")

    return {
        "paths": paths,
        "stats": {
            "crus": asdict(crus_stats),
            "sn":   asdict(sn_stats),
            "sn_l": asdict(l_stats),
            "sn_r": asdict(r_stats),
        },
    }


def kde_mode(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 10:
        raise ValueError(f"Мало значений: {values.size}")
    bins, counts = np.unique(values, return_counts=True)
    threshold = bins[counts > counts.mean()]
    if threshold.size > 0:
        vals_cut = values[(values > threshold.min()) & (values < threshold.max())]
        if vals_cut.size >= 10:
            values = vals_cut
    kernel = gaussian_kde(values, bw_method="scott")
    x0 = float(values[np.argmax(kernel.pdf(values))])
    res = minimize(lambda x: -float(kernel(x)[0]), x0=x0,
                    bounds=[[values.min(), values.max()]])
    return float(res.x[0])


def compute_nm_cnr(nm_mri_path: Path, masks: dict, out_dir: Path) -> dict:
    log("NM-CNR (KCL-конвенция)", "step")
    log(f"  NM-MRI: {nm_mri_path.name}")

    nm, _, header, _ = load_canonical(nm_mri_path)
    ref, _, _, _ = load_canonical(masks["paths"]["CrusCerebri"])
    l_, _, _, _ = load_canonical(masks["paths"]["SN_VTA_L"])
    r_, _, _, _ = load_canonical(masks["paths"]["SN_VTA_R"])

    ref_mask = ref > 0.5
    roi_l = l_ > 0.5
    roi_r = r_ > 0.5

    if ref_mask.sum() < 20:
        log(f"  Маска Crus мала: {ref_mask.sum()}", "warn")
        return {}

    sn_signal = float(np.mean(nm[roi_l | roi_r]))
    crus_signal = float(np.mean(nm[ref_mask]))
    log(f"  Сигнал SN:   {sn_signal:.2f}")
    log(f"  Сигнал Crus: {crus_signal:.2f}")
    log(f"  SN/Crus:     {sn_signal / crus_signal:.3f}")

    cc_mode = kde_mode(nm[ref_mask])
    cnr_map = (nm - cc_mode) / cc_mode

    cnr_l = float(np.mean(cnr_map[roi_l]))
    cnr_r = float(np.mean(cnr_map[roi_r]))
    cnr_mean = 0.5 * (cnr_l + cnr_r)

    log(f"  Mode(Crus) = {cc_mode:.4f}")
    log(f"  CNR = {cnr_mean * 100:+.2f}%  "
        f"[L={cnr_l * 100:+.2f}%, R={cnr_r * 100:+.2f}%]")

    if cnr_mean > 0:
        log(f"  ✅ SN ярче Crus", "ok")
    else:
        log(f"  ⚠ SN темнее Crus", "warn")

    mag_img = nib.load(str(nm_mri_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(cnr_map.astype(np.float32),
                              mag_img.affine, header),
             str(out_dir / "nm_cnr_map.nii.gz"))

    return {
        "nm_mri_file": nm_mri_path.name,
        "mode_crus": cc_mode,
        "signal_sn_mean": sn_signal,
        "signal_crus_mean": crus_signal,
        "signal_ratio": sn_signal / crus_signal,
        "cnr_mean": cnr_mean, "cnr_left": cnr_l, "cnr_right": cnr_r,
        "cnr_mean_pct": cnr_mean * 100,
        "cnr_left_pct": cnr_l * 100,
        "cnr_right_pct": cnr_r * 100,
    }


def _roi_metrics(qsm, mask):
    vals = qsm[mask]
    return {
        "mean":   float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std":    float(np.std(vals)),
        "p95":    float(np.percentile(vals, 95)),
        "max":    float(np.max(vals)),
        "voxels": int(mask.sum()),
    }


def compute_qsm_in_roi(qsm_path: Path, masks: dict) -> dict:
    log("QSM в SN-VTA", "step")

    if not qsm_path.exists():
        log(f"  QSM не найден: {qsm_path.name}", "warn")
        return {}

    # Проверяем, есть ли QSM-маски со сдвигом
    work = masks["paths"]["SN_VTA_L"].parent
    sn_l_qsm = work / "SN_VTA_L_qsm.nii.gz"
    sn_r_qsm = work / "SN_VTA_R_qsm.nii.gz"

    if sn_l_qsm.exists() and sn_r_qsm.exists():
        log(f"  Использую QSM-маски (со сдвигом)")
        l_path, r_path = sn_l_qsm, sn_r_qsm
    else:
        log(f"  Использую обычные маски SN")
        l_path = masks["paths"]["SN_VTA_L"]
        r_path = masks["paths"]["SN_VTA_R"]

    qsm, qsm_aff, _, _ = load_canonical(qsm_path)
    l_, l_aff, _, _ = load_canonical(l_path)
    r_, _, _, _ = load_canonical(r_path)
    c_, _, _, _ = load_canonical(masks["paths"]["CrusCerebri"])

    roi_l = l_ > 0.5
    roi_r = r_ > 0.5
    ref_mask = c_ > 0.5

    if qsm.shape != roi_l.shape:
        log(f"  ⚠ Форма QSM {qsm.shape} != маски {roi_l.shape}", "err")
        return {}

    if not np.allclose(qsm_aff, l_aff, atol=0.01):
        log("  ⚠ Affine не совпадают", "warn")
    else:
        log("  ✅ Affine совпадают")

    m_l = _roi_metrics(qsm, roi_l)
    m_r = _roi_metrics(qsm, roi_r)
    m_crus = _roi_metrics(qsm, ref_mask)

    def avg(key):
        return 0.5 * (m_l[key] + m_r[key])

    metrics_avg = {k: avg(k) for k in ["mean", "median", "p95", "max"]}
    chi_crus = m_crus["mean"]

    sn_cc = {k: (metrics_avg[k] / chi_crus if chi_crus else 0)
             for k in ["mean", "median", "p95", "max"]}

    log(f"  χ SN-L: mean={m_l['mean']:+.4f}  p95={m_l['p95']:+.4f}  "
        f"max={m_l['max']:+.4f}")
    log(f"  χ SN-R: mean={m_r['mean']:+.4f}  p95={m_r['p95']:+.4f}  "
        f"max={m_r['max']:+.4f}")
    log(f"  χ SN avg: mean={metrics_avg['mean']:+.4f}  "
        f"p95={metrics_avg['p95']:+.4f}  max={metrics_avg['max']:+.4f}")
    log(f"  χ Crus:  {chi_crus:+.4f} ppm")
    log(f"  SN/CC:   mean={sn_cc['mean']:.3f}  p95={sn_cc['p95']:.3f}")

    return {
        "sn_l": m_l, "sn_r": m_r, "sn_avg": metrics_avg, "crus": m_crus,
        "chi_mean":   metrics_avg["mean"],
        "chi_median": metrics_avg["median"],
        "chi_p95":    metrics_avg["p95"],
        "chi_max":    metrics_avg["max"],
        "chi_left":   m_l["mean"],
        "chi_right":  m_r["mean"],
        "chi_reference": chi_crus,
        "sn_cc_ratio":        sn_cc["mean"],
        "sn_cc_ratio_median": sn_cc["median"],
        "sn_cc_ratio_p95":    sn_cc["p95"],
        "sn_cc_ratio_max":    sn_cc["max"],
    }

def _compute_robust_safe(args, work: Path, masks: dict) -> dict:
    """
    Обёртка: robust-метод с graceful fallback.
    Возвращает {} если модуль недоступен или упал.
    """
    try:
        from neuromelanin.cnr_robust import compute_nm_cnr_robust
    except ImportError:
        log("  cnr_robust.py не найден — robust пропущен")
        return {}

    try:
        return compute_nm_cnr_robust(
            nm_mri_path=args.nm_mri,
            sn_l_path=masks["paths"]["SN_VTA_L"],
            sn_r_path=masks["paths"]["SN_VTA_R"],
            crus_path=masks["paths"]["CrusCerebri"],
            trim=0.1,
            min_sn_voxels=5,
            min_crus_voxels=20,
            verbose=True,
        )
    except Exception as e:
        log(f"  ⚠ robust упал: {e}", "warn")
        return {}


def save_montage(qsm_path: Path, out_dir: Path, n_slices=9):
    if not HAS_MPL or not qsm_path.exists():
        return None

    log("Монтаж (PNG)", "step")

    qsm, _, _, _ = load_canonical(qsm_path)
    combined, _, _, _ = load_canonical(out_dir / "all_masks_combined.nii.gz")
    combined = combined.astype(int)

    coords = np.argwhere(combined == 2)
    if len(coords) == 0:
        return None

    z_center = int(coords[:, 2].mean())
    z_range = np.linspace(max(0, z_center - 8),
                           min(qsm.shape[2] - 1, z_center + 8),
                           n_slices).astype(int)

    vmin = float(np.percentile(qsm[combined > 0], 5))
    vmax = float(np.percentile(qsm[combined > 0], 95))

    n_cols = 3
    n_rows = int(np.ceil(n_slices / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 4 * n_rows), dpi=110)
    axes = np.atleast_2d(axes).ravel()

    for i, z in enumerate(z_range):
        ax = axes[i]
        ax.imshow(qsm[:, :, z].T, cmap="gray", vmin=vmin, vmax=vmax,
                   origin="lower", interpolation="nearest")
        overlay = np.zeros((*combined[:, :, z].T.shape, 4), dtype=np.float32)
        s = combined[:, :, z].T
        overlay[s == 1] = [1, 0, 0, 0.55]
        overlay[s == 2] = [0, 1, 0, 0.55]
        overlay[s == 3] = [0, 0.5, 1, 0.45]
        ax.imshow(overlay, origin="lower", interpolation="nearest")
        ax.set_title(f"z = {z}", fontsize=10)
        ax.axis("off")

    for i in range(len(z_range), len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"SN-VTA on QSM [{vmin:+.3f}, {vmax:+.3f}] ppm", fontsize=12)
    plt.tight_layout()

    out = out_dir / "qc_montage.png"
    plt.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    log(f"  ✅ → {out.name}", "ok")
    return out


def _main_impl():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--qsm", type=Path, required=True)
    p.add_argument("--mag-mean", type=Path, required=True)
    p.add_argument("--nm-mri", type=Path, required=True)
    p.add_argument("--atlas-native", type=Path, required=True)
    p.add_argument("--reg-meta", type=Path, required=True)
    p.add_argument("--dicom-dir", type=Path, default=None)
    p.add_argument("--field-strength", type=float, default=None)
    p.add_argument("--nm-sequence", type=str, default="auto",
                   choices=["auto", "tse", "mtc_gre", "gre"])
    p.add_argument("--no-montage", action="store_true")
    args = p.parse_args()

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    try:
        reg_meta = json.loads(args.reg_meta.read_text())
        reg_type = (f"{reg_meta.get('type_t1_mni', '?')} + "
                    f"{reg_meta.get('type_nm_t1', '?')}")
    except Exception:
        reg_meta = {}
        reg_type = "unknown"

    # --- Демография ---
    log("Демография пациента", "step")
    data_dir = args.nm_mri.parent
    dicom_dir = args.dicom_dir if args.dicom_dir else (data_dir / "dicom")

    patient = PatientInfo()
    if HAS_DEMOGRAPHICS:
        patient = get_patient_info(
            dicom_dir if dicom_dir.exists() else data_dir
        )

    if patient.field_strength is None and args.field_strength is not None:
        patient.field_strength = args.field_strength
    if patient.field_strength is None:
        patient.field_strength = 3.0
        log("⚠ B0 не определён — считаю как 3.0T", "warn")

    log(f"  Пациент: {patient.summary()}", "ok" if patient.age else "warn")

    # --- Определение типа последовательности ---
    nm_seq = args.nm_sequence
    if nm_seq == "auto" and HAS_DEMOGRAPHICS:
        json_path = args.nm_mri.with_suffix(".json")
        nm_seq = detect_sequence_type(json_path)
        log(f"  Последовательность (auto): {nm_seq}")
    elif nm_seq == "auto":
        nm_seq = "tse"
        log(f"  Последовательность: fallback на tse")

    if nm_seq == "gre":
        log("  ⚠ GRE без MT — NM-CNR не будет иметь NM-контраста!", "warn")

    # --- Маски ---
    masks = extract_masks(work, work)

    # --- NM-CNR ---
    nm_result = compute_nm_cnr(args.nm_mri, masks, work)

    # --- Сравнение с нормой ---
    comparison = None
    if nm_result and patient.age is not None and HAS_DEMOGRAPHICS:
        log("Сравнение CNR с нормой", "step")
        try:
            comparison = compare_with_normative(
                nm_result["cnr_mean_pct"], patient, sequence=nm_seq,
            )
            log(f"  Поле:            {comparison.field_used:.1f}T")
            log(f"  Последовательность: {comparison.sequence_used}")
            log(f"  Источник нормы:  {comparison.normative_source}")
            log(f"  Ожидаемый CNR:   {comparison.expected_cnr_pct:+.2f}%")
            log(f"  Z-score:         {comparison.z_score:+.2f}")
            log(f"  Интерпретация:   {comparison.interpretation}")
            if comparison.warning:
                log(f"  ⚠ {comparison.warning}", "warn")
        except Exception as e:
            log(f"  Не удалось сравнить: {e}", "warn")

    # --- QSM в SN ---
    qsm_result = compute_qsm_in_roi(args.qsm, masks)

    # --- NM-CNR robust (диагностический) ---
    log("Robust NM-CNR (slice-wise + ipsilateral)", "step")
    nm_result_robust = _compute_robust_safe(args, work, masks)

    comparison_robust = None
    if nm_result_robust and patient.age is not None and HAS_DEMOGRAPHICS:
        try:
            comparison_robust = compare_with_normative(
                nm_result_robust["cnr_mean_pct"],
                patient, sequence=nm_seq,
            )
            log(f"  Robust: CNR={nm_result_robust['cnr_mean_pct']:+.2f}%  "
                f"z={comparison_robust.z_score:+.2f}  "
                f"({comparison_robust.interpretation})")
        except Exception as e:
            log(f"  ⚠ robust сравнение упало: {e}", "warn")

    if nm_result and nm_result_robust:
        base = nm_result["cnr_mean"]
        rob = nm_result_robust["cnr_mean"]
        d = abs(base - rob)
        rel = 100 * d / abs(base) if base else 0.0
        log(f"  Δ(base vs robust) = {d:.4f} ({rel:.1f}%)")
        if rel > 20:
            log(f"  ⚠ >20% — B1- или L/R-неоднородности", "warn")
        elif rel > 10:
            log(f"  ⚠ 10–20% — посмотрите QC-монтаж", "warn")
        else:
            log(f"  ✅ Методы согласуются")

    # --- Монтаж ---
    montage_path = None
    if not args.no_montage:
        montage_path = save_montage(args.qsm, work)

    # --- Отчёт ---
    report = {
        "timestamp": datetime.now().isoformat(),
        "patient": asdict(patient) if HAS_DEMOGRAPHICS else {},
        "nm_sequence": nm_seq,
        "registration": reg_meta,
        "masks": masks["stats"],
        "nm_cnr": nm_result,
        "nm_cnr_robust": nm_result_robust,
        "comparison_with_normative": (
            asdict(comparison) if comparison else None
        ),
        "comparison_with_normative_robust": (
            asdict(comparison_robust) if comparison_robust else None
        ),
        "qsm_in_sn": qsm_result,
    }
    report_path = work / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    # ==========================================================================
    # ИТОГ
    # ==========================================================================
    print("\n" + "=" * 65)
    print("  ИТОГ")
    print("=" * 65)
    print(f"\n  Пациент: {patient.summary()}")
    print(f"  Поле:    {patient.field_strength:.1f}T")
    print(f"  Последовательность NM: {nm_seq}")
    print(f"  Регистрация: {reg_type}")

    if nm_result:
        print(f"\n  🅰️  NM-CNR:")
        print(f"      Signal SN/Crus = {nm_result['signal_ratio']:.3f}")
        print(f"      CNR mean = {nm_result['cnr_mean_pct']:+.2f}%   "
              f"[L={nm_result['cnr_left_pct']:+.2f}%, "
              f"R={nm_result['cnr_right_pct']:+.2f}%]")

    if nm_result_robust:
        d = abs(nm_result["cnr_mean"] - nm_result_robust["cnr_mean"])
        rel = (100 * d / abs(nm_result["cnr_mean"])
               if nm_result.get("cnr_mean") else 0)
        print(f"      Robust CNR = {nm_result_robust['cnr_mean_pct']:+.2f}% "
              f"(Δ={rel:.1f}%)")

    if comparison:
        print(f"\n  🧬 Сравнение с нормой ({comparison.field_used:.1f}T, "
              f"{comparison.sequence_used}):")
        print(f"      Ожидаемый CNR:   {comparison.expected_cnr_pct:+.2f}%")
        print(f"      Z-score:         {comparison.z_score:+.2f}")
        print(f"      ⭐ Интерпретация: {comparison.interpretation}")
        if comparison.warning:
            print(f"      ⚠ {comparison.warning}")

    if qsm_result:
        print(f"\n  🅱️  QSM в SN-VTA:")
        print(f"      ┌─ Метрика ──── SN-VTA ──── SN/CC ─┐")
        print(f"      │ mean       {qsm_result['chi_mean']:+.4f}     "
              f"{qsm_result['sn_cc_ratio']:.3f} │")
        print(f"      │ p95 ★      {qsm_result['chi_p95']:+.4f}     "
              f"{qsm_result['sn_cc_ratio_p95']:.3f} │")
        print(f"      │ max        {qsm_result['chi_max']:+.4f}     "
              f"{qsm_result['sn_cc_ratio_max']:.3f} │")
        print(f"      └────────────────────────────────┘")
        print(f"      χ Crus = {qsm_result['chi_reference']:+.4f} ppm")

    print(f"\n  Файлы:")
    print(f"      Отчёт:   {report_path}")
    if montage_path:
        print(f"      Монтаж:  {montage_path}")
    print()


def main():
    try:
        _main_impl()
    except Exception:
        print("\n" + "=" * 65, file=sys.stderr)
        print("  ❌ TRACEBACK (analyze.py)", file=sys.stderr)
        print("=" * 65, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()