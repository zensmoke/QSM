"""
Прямое измерение χ вдоль вертикальной и горизонтальной линий,
проходящих через SN. Покажет, есть ли пик в SN.
"""
import nibabel as nib
import numpy as np
from pathlib import Path

OUT = Path("/Users/nikitamyasov/3782185 ласточка/output")
NM = OUT / "nm_analysis"

# Загрузка
qsm_img = nib.load(str(OUT / "qsm.nii.gz"), mmap=False)
qsm_img = nib.as_closest_canonical(qsm_img)
qsm = qsm_img.get_fdata(dtype=np.float32)

sn_l_img = nib.load(str(NM / "SN_VTA_L_native.nii.gz"), mmap=False)
sn_l_img = nib.as_closest_canonical(sn_l_img)
sn_l = sn_l_img.get_fdata() > 0.5

crus_img = nib.load(str(NM / "CrusCerebri_native.nii.gz"), mmap=False)
crus_img = nib.as_closest_canonical(crus_img)
crus = crus_img.get_fdata() > 0.5

# Центр SN
coords = np.argwhere(sn_l)
if len(coords) == 0:
    print("SN-L пуст!")
    exit(1)

cx, cy, cz = coords.mean(axis=0).astype(int)
print(f"Центр SN-L: X={cx}, Y={cy}, Z={cz}")

# Горизонтальный профиль через центр SN (по X)
print(f"\nГоризонтальный профиль через SN-L (Y={cy}, Z={cz}):")
print(f"{'X':>5} {'χ (ppm)':>12} {'SN-L':>6} {'Crus':>6}")
for dx in range(-30, 31, 3):
    x = cx + dx
    chi = qsm[x, cy, cz]
    in_sn = "  ✓" if sn_l[x, cy, cz] else "   "
    in_crus = "  ✓" if crus[x, cy, cz] else "   "
    print(f"{x:>5} {chi:+12.4f} {in_sn:>6} {in_crus:>6}")

# Статистика
print(f"\n--- Статистика по ROI ---")
print(f"SN-L:  mean={qsm[sn_l].mean():+.4f}, median={np.median(qsm[sn_l]):+.4f}, "
      f"max={qsm[sn_l].max():+.4f}")
print(f"Crus:  mean={qsm[crus].mean():+.4f}, median={np.median(qsm[crus]):+.4f}, "
      f"max={qsm[crus].max():+.4f}")

# Какой процент вокселей SN ярче порога 0.04 ppm?
print(f"\nВ SN-L вокселей с χ > 0.04 ppm: {(qsm[sn_l] > 0.04).sum()}")
print(f"В SN-L вокселей с χ > 0.05 ppm: {(qsm[sn_l] > 0.05).sum()}")
print(f"В SN-L вокселей с χ > 0.06 ppm: {(qsm[sn_l] > 0.06).sum()}")

# 95-й процентиль
print(f"\n95-й процентиль χ в SN-L: {np.percentile(qsm[sn_l], 95):+.4f} ppm")
print(f"95-й процентиль χ в Crus: {np.percentile(qsm[crus], 95):+.4f} ppm")