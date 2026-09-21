# check_sn_l_artifact.py
from pathlib import Path
import nibabel as nib
import numpy as np

QSM = Path("/Users/nikitamyasov/3782185 ласточка/output/qsm.nii.gz")
NM = Path("/Users/nikitamyasov/3782185 ласточка/output/nm_analysis")

qsm = nib.load(str(QSM), mmap=False)
qsm = nib.as_closest_canonical(qsm).get_fdata()

sn_l = nib.load(str(NM / "SN_VTA_L_qsm.nii.gz"), mmap=False)
sn_l = nib.as_closest_canonical(sn_l).get_fdata() > 0.5

# Профиль χ по X через SN-L
coords = np.argwhere(sn_l)
cx, cy, cz = coords.mean(axis=0).astype(int)

print(f"Профиль χ по X через SN-L (центр Y={cy}, Z={cz}):")
for dx in range(-30, 31, 5):
    x = cx + dx
    if 0 <= x < qsm.shape[0]:
        in_mask = "✓" if sn_l[x, cy, cz] else " "
        print(f"  X={x:3d} ({dx:+3d}): {qsm[x, cy, cz]:+.4f}  {in_mask}")