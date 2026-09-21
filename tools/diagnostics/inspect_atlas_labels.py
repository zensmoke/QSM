# inspect_atlas_labels.py
import nibabel as nib
import numpy as np
from pathlib import Path

atlas_path = Path("/KCL/templates/midbrain_atlas_space-MNI152NLin2009cSym.nii.gz")
img = nib.load(str(atlas_path))
data = np.asarray(img.dataobj)

print(f"Форма: {data.shape}")
print(f"dtype: {data.dtype}")
print(f"Уникальные значения: {np.unique(data.astype(int))}")
print()

# Посчитать объём каждого label'а
unique_labels = np.unique(data.astype(int))
for lbl in unique_labels:
    if lbl == 0:
        continue
    count = (data == lbl).sum()
    print(f"Label {lbl}: {count} вокселей ({count * img.header.get_zooms()[0]**3:.2f} мм³)")