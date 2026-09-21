"""
Тест нашего модуля нейромеланина на готовых данных KCL (sub-001).
Сверяем результат с официальным NM_CNR_SNVTA_Results.xlsx.
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# --- Пути ---
KCL = Path("/KCL")
SUB = "sub-001"

NM_MRI = KCL / f"nifti/{SUB}/anat/{SUB}_NM.nii.gz"
ATLAS_NATIVE = KCL / f"output/{SUB}/anat/{SUB}_midbrain_atlas_space-NM.nii.gz"
KCL_CNR_RESULTS = KCL / "results/NM_CNR_SNVTA_Results.xlsx"

# --- Извлечь маски в нативном пространстве ---
atlas = nib.load(str(ATLAS_NATIVE))
atlas_data = np.asarray(atlas.dataobj).astype(int)

print(f"Атлас: {ATLAS_NATIVE.name}")
print(f"Форма: {atlas_data.shape}")
print(f"Уникальные метки: {np.unique(atlas_data)}")

crus_mask = (atlas_data == 1)
sn_mask = (atlas_data == 2)

print(f"\nCrus Cerebri: {crus_mask.sum()} вокселей")
print(f"SN-VTA:       {sn_mask.sum()} вокселей")

# Разделение SN-VTA на L/R
coords = np.argwhere(sn_mask)
center_x = coords[:, 0].mean()
sn_left = np.zeros_like(sn_mask)
sn_right = np.zeros_like(sn_mask)
for x, y, z in coords:
    if x < center_x:
        sn_left[x, y, z] = 1
    else:
        sn_right[x, y, z] = 1

print(f"  SN-VTA Left:  {sn_left.sum()} вокселей")
print(f"  SN-VTA Right: {sn_right.sum()} вокселей")

# --- Сохранить маски ---
out_dir = KCL / "roi_extracted"
out_dir.mkdir(exist_ok=True)

for name, m in [("CrusCerebri", crus_mask),
                 ("SN_VTA", sn_mask),
                 ("SN_VTA_L", sn_left),
                 ("SN_VTA_R", sn_right)]:
    path = out_dir / f"{name}_native.nii.gz"
    nib.save(nib.Nifti1Image(m.astype(np.uint8), atlas.affine, atlas.header),
             str(path))
    print(f"  → {path}")

# --- Запустить наш модуль ---
sys.path.insert(0, str(Path(__file__).parent))

from neuromelanin.cnr import calculate_cnr_kcl

result = calculate_cnr_kcl(
    nm_mri_path=NM_MRI,
    roi_left_path=out_dir / "SN_VTA_L_native.nii.gz",
    roi_right_path=out_dir / "SN_VTA_R_native.nii.gz",
    reference_mask_path=out_dir / "CrusCerebri_native.nii.gz",
    output_dir=KCL / "our_test_output",
    verbose=True,
)

# --- Сравнить с KCL ---
print("\n" + "=" * 62)
print("  СРАВНЕНИЕ С ОФИЦИАЛЬНЫМИ РЕЗУЛЬТАТАМИ KCL")
print("=" * 62)

try:
    import pandas as pd
    kcl_df = pd.read_excel(KCL_CNR_RESULTS)
    kcl_sub = kcl_df[kcl_df["Subject ID"] == SUB]

    if not kcl_sub.empty:
        kcl_mean = float(kcl_sub.iloc[0]["Mean"])
        kcl_std = float(kcl_sub.iloc[0]["Standard Deviation"])

        our_mean = result["mean"]["cnr"]
        our_std = result["mean"]["cnr_std"]

        print(f"  KCL:  Mean = {kcl_mean:.6f}, SD = {kcl_std:.6f}")
        print(f"  Наши: Mean = {our_mean:.6f}, SD = {our_std:.6f}")
        print(f"  Δ Mean = {abs(kcl_mean - our_mean):.6f}")
        print(f"  Δ SD   = {abs(kcl_std - our_std):.6f}")

        if abs(kcl_mean - our_mean) < 1e-4 and abs(kcl_std - our_std) < 1e-4:
            print("\n  ✅ ПОЛНОЕ СОВПАДЕНИЕ")
        elif abs(kcl_mean - our_mean) < 1e-3:
            print("\n  ✅ Отличное совпадение (< 0.001)")
        else:
            print("\n  ⚠️  Расхождение всё ещё есть")
            print("     Проверьте импорт KCL refined_mode_estimation")
    else:
        print(f"  Subject {SUB} не найден в {KCL_CNR_RESULTS.name}")

except ImportError as e:
    print(f"  ❌ Не установлен pandas/openpyxl: {e}")
    print(f"     Установите: pip install pandas openpyxl")
except Exception as e:
    print(f"  ❌ Ошибка при сравнении: {e}")

print("=" * 62)