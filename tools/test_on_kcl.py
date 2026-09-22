"""
Тест NM-модуля на готовых данных KCL (sub-001).

Сравниваем ТРИ числа с официальным NM_CNR_SNVTA_Results.xlsx:
  1. KCL official  — эталон
  2. base          — neuromelanin.analyze.compute_nm_cnr
                     (KCL-конвенция: mode(Crus) по всему объёму)
  3. robust        — neuromelanin.cnr_robust.compute_nm_cnr_robust
                     (slice-wise + ipsilateral + trimmed mean)
"""
import sys
from pathlib import Path

# Корень проекта = родитель tools/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np


# =============================================================================
# ПУТИ
# =============================================================================
KCL = Path("/Users/nikitamyasov/PycharmProjects/QSM/KCL")
SUB = "sub-001"

NM_MRI       = KCL / f"nifti/{SUB}/anat/{SUB}_NM.nii.gz"
ATLAS_NATIVE = KCL / f"output/{SUB}/anat/{SUB}_midbrain_atlas_space-NM.nii.gz"
KCL_CNR_XLSX = KCL / "results/NM_CNR_SNVTA_Results.xlsx"

ROI_OUT = KCL / "roi_extracted"
OUT_DIR = KCL / "our_test_output"


# =============================================================================
# 1. МАСКИ
# =============================================================================
def extract_masks_from_atlas() -> dict:
    print("=" * 62)
    print("  ШАГ 1: Извлечение масок из атласа KCL")
    print("=" * 62)

    if not ATLAS_NATIVE.exists():
        raise FileNotFoundError(f"Атлас не найден: {ATLAS_NATIVE}")

    atlas_img = nib.load(str(ATLAS_NATIVE))
    atlas_data = np.asarray(atlas_img.dataobj).astype(int)

    print(f"  Атлас: {ATLAS_NATIVE.name}")
    print(f"  Форма: {atlas_data.shape}")
    print(f"  Метки: {sorted(np.unique(atlas_data).tolist())}")

    crus_mask = (atlas_data == 1)
    sn_mask   = (atlas_data == 2)
    if crus_mask.sum() == 0 or sn_mask.sum() == 0:
        raise RuntimeError(
            f"Пустые маски: Crus={crus_mask.sum()}, SN={sn_mask.sum()}"
        )

    # L/R split по X-центроиду
    def split_lr(mask):
        coords = np.argwhere(mask)
        cx = float(coords[:, 0].mean())
        idx = np.indices(mask.shape)[0]
        return (mask & (idx < cx)).astype(np.uint8), \
               (mask & (idx >= cx)).astype(np.uint8)

    sn_l, sn_r = split_lr(sn_mask)

    print(f"\n  Воксели:")
    print(f"    Crus:    {int(crus_mask.sum())}")
    print(f"    SN-VTA:  {int(sn_mask.sum())}")
    print(f"    SN-L/R:  {int(sn_l.sum())} / {int(sn_r.sum())}")

    ROI_OUT.mkdir(parents=True, exist_ok=True)

    def save(name, arr):
        path = ROI_OUT / f"{name}_native.nii.gz"
        nib.save(
            nib.Nifti1Image(arr.astype(np.uint8),
                            atlas_img.affine, atlas_img.header),
            str(path),
        )
        return path

    paths = {
        "CrusCerebri": save("CrusCerebri", crus_mask.astype(np.uint8)),
        "SN_VTA":      save("SN_VTA",      sn_mask.astype(np.uint8)),
        "SN_VTA_L":    save("SN_VTA_L",    sn_l),
        "SN_VTA_R":    save("SN_VTA_R",    sn_r),
    }
    print(f"\n  Маски: {ROI_OUT}")
    return paths


# =============================================================================
# 2. БАЗОВЫЙ МЕТОД — neuromelanin.analyze.compute_nm_cnr
# =============================================================================
def run_baseline(mask_paths: dict) -> dict:
    print("\n" + "=" * 62)
    print("  ШАГ 2: Базовый метод (neuromelanin.analyze.compute_nm_cnr)")
    print("=" * 62)

    from neuromelanin.analyze import compute_nm_cnr

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # compute_nm_cnr ожидает dict с ключом "paths"
    masks = {"paths": {
        "CrusCerebri": mask_paths["CrusCerebri"],
        "SN_VTA":      mask_paths["SN_VTA"],
        "SN_VTA_L":    mask_paths["SN_VTA_L"],
        "SN_VTA_R":    mask_paths["SN_VTA_R"],
    }}

    result = compute_nm_cnr(NM_MRI, masks, OUT_DIR)

    if not result:
        print("  ⚠ compute_nm_cnr вернул пустой dict")
        return {}

    print(f"\n  Базовый результат:")
    print(f"    cnr_mean     = {result['cnr_mean']:.6f}")
    print(f"    cnr_left     = {result['cnr_left']:.6f}")
    print(f"    cnr_right    = {result['cnr_right']:.6f}")
    return result


# =============================================================================
# 3. ROBUST МЕТОД
# =============================================================================
def run_robust(mask_paths: dict) -> dict:
    print("\n" + "=" * 62)
    print("  ШАГ 3: Robust (slice-wise + ipsilateral + trimmed mean)")
    print("=" * 62)

    try:
        from neuromelanin.cnr_robust import compute_nm_cnr_robust
    except ImportError as e:
        print(f"  ❌ cnr_robust.py не найден: {e}")
        return {}

    result = compute_nm_cnr_robust(
        nm_mri_path=NM_MRI,
        sn_l_path=mask_paths["SN_VTA_L"],
        sn_r_path=mask_paths["SN_VTA_R"],
        crus_path=mask_paths["CrusCerebri"],
        trim=0.1,
        min_sn_voxels=5,
        min_crus_voxels=20,
        verbose=True,
    )

    print(f"\n  Per-slice CNR-L:")
    for z, v in zip(result["z_slices_left"], result["cnr_left_slices"]):
        print(f"    z = {z:3d}   CNR = {v*100:+7.2f}%")
    print(f"  Per-slice CNR-R:")
    for z, v in zip(result["z_slices_right"], result["cnr_right_slices"]):
        print(f"    z = {z:3d}   CNR = {v*100:+7.2f}%")

    return result


# =============================================================================
# 4. СРАВНЕНИЕ
# =============================================================================
def compare_all(baseline: dict, robust: dict) -> None:
    print("\n" + "=" * 62)
    print("  СРАВНЕНИЕ С KCL")
    print("=" * 62)

    try:
        import pandas as pd
    except ImportError:
        print("  ❌ pip install pandas openpyxl")
        return

    try:
        df = pd.read_excel(KCL_CNR_XLSX)
        row = df[df["Subject ID"] == SUB]
        if row.empty:
            print(f"  {SUB} не найден в {KCL_CNR_XLSX.name}")
            return
        kcl_mean = float(row.iloc[0]["Mean"])
        kcl_std  = float(row.iloc[0]["Standard Deviation"])
    except Exception as e:
        print(f"  ❌ {e}")
        return

    base_mean = baseline.get("cnr_mean", float("nan"))
    rb_mean   = robust.get("cnr_mean",  float("nan"))

    print(f"\n  {'Метод':<28} {'Mean':>14} {'Δ vs KCL':>14}")
    print(f"  {'-'*28} {'-'*14} {'-'*14}")
    print(f"  {'KCL official':<28} {kcl_mean:>14.6f} {'—':>14}")
    print(f"  {'base (mode global)':<28} {base_mean:>14.6f} "
          f"{abs(kcl_mean - base_mean):>14.6f}")
    print(f"  {'robust (slice+ipsi)':<28} {rb_mean:>14.6f} "
          f"{abs(kcl_mean - rb_mean):>14.6f}")

    if not (np.isnan(base_mean) or np.isnan(rb_mean)):
        d = abs(base_mean - rb_mean)
        rel = 100 * d / abs(base_mean) if base_mean else 0
        print(f"\n  Δ(base vs robust) = {d:.6f}  ({rel:.1f}%)")
        if rel > 20:
            print(f"  ⚠ > 20% — вероятны B1- или L/R-неоднородности.")
        elif rel > 10:
            print(f"  ⚠ 10–20% — посмотрите per-slice CNR выше.")
        else:
            print(f"  ✅ Методы согласуются (< 10%).")

    def verdict(x):
        if np.isnan(x): return "нет данных"
        if x < 1e-4:    return "✅ ПОЛНОЕ СОВПАДЕНИЕ"
        if x < 1e-3:    return "✅ Отличное"
        if x < 5e-3:    return "🟡 Приемлемо"
        return "⚠️  Расхождение"

    print(f"\n  Вердикт:")
    print(f"    base   vs KCL: {verdict(abs(kcl_mean - base_mean))}")
    print(f"    robust vs KCL: {verdict(abs(kcl_mean - rb_mean))}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"\n  ТЕСТ NM-CNR НА {SUB}")
    print(f"  NM-MRI: {NM_MRI}")
    print(f"  Атлас:  {ATLAS_NATIVE}")
    print(f"  Эталон: {KCL_CNR_XLSX}")

    masks = extract_masks_from_atlas()

    baseline = {}
    try:
        baseline = run_baseline(masks)
    except Exception as e:
        print(f"\n  ❌ base: {e}")
        import traceback; traceback.print_exc()

    robust = {}
    try:
        robust = run_robust(masks)
    except Exception as e:
        print(f"\n  ❌ robust: {e}")
        import traceback; traceback.print_exc()

    compare_all(baseline, robust)
    print("\n" + "=" * 62)


if __name__ == "__main__":
    main()