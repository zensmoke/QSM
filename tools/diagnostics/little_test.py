import nibabel as nib
import numpy as np

atlas = nib.load("/KCL/templates/midbrain_atlas_space-MNI152NLin2009cSym.nii.gz")
data = np.asarray(atlas.dataobj).astype(int)

for label in [1, 2]:
    coords = np.argwhere(data == label)
    if len(coords) == 0:
        print(f"Label {label}: пусто")
        continue
    center = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0)
    print(f"Label {label}: {len(coords)} вокселей")
    print(f"  Центр (X,Y,Z): {center}")
    print(f"  Размер (мм):   {extent}")