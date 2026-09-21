"""
Объединяет SN_VTA_L, SN_VTA_R и CrusCerebri в один файл
с разными числовыми метками для отображения в ITK-SNAP.
Метки:
  1 = SN-VTA левая (красная)
  2 = SN-VTA правая (зелёная)
  3 = Crus Cerebri (синяя)
"""
from pathlib import Path
import nibabel as nib
import numpy as np

NM_OUT = Path("/Users/nikitamyasov/3782185 ласточка/output/nm_analysis")

# Загрузка масок
sn_l = np.asarray(nib.load(str(NM_OUT / "SN_VTA_L_native.nii.gz")).dataobj) > 0.5
sn_r = np.asarray(nib.load(str(NM_OUT / "SN_VTA_R_native.nii.gz")).dataobj) > 0.5
crus = np.asarray(nib.load(str(NM_OUT / "CrusCerebri_native.nii.gz")).dataobj) > 0.5

# Референс (для affine)
ref_img = nib.load(str(NM_OUT / "SN_VTA_L_native.nii.gz"))

# Создаём combined: 0 = фон, 1 = SN_L, 2 = SN_R, 3 = Crus
combined = np.zeros(sn_l.shape, dtype=np.uint8)
combined[crus] = 3
combined[sn_l] = 1
combined[sn_r] = 2

out_path = NM_OUT / "all_masks_combined.nii.gz"
nib.save(nib.Nifti1Image(combined, ref_img.affine, ref_img.header),
         str(out_path))

print(f"Сохранено: {out_path}")
print(f"  SN-L (label 1): {sn_l.sum()} вокселей")
print(f"  SN-R (label 2): {sn_r.sum()} вокселей")
print(f"  Crus (label 3): {crus.sum()} вокселей")