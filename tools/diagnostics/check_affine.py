# check_affine.py
from pathlib import Path
import nibabel as nib
import numpy as np

QSM = Path("/Users/nikitamyasov/3782185 ласточка/output/qsm.nii.gz")
MASK_DIR = Path("/Users/nikitamyasov/3782185 ласточка/output/nm_analysis")
BRAIN_MASK = Path("/Users/nikitamyasov/3782185 ласточка/output/brain_mask.nii.gz")


def header_info(path, name):
    img = nib.load(str(path), mmap=False)
    print(f"\n{name}:")
    print(f"  shape:   {img.shape}")
    print(f"  dtype:   {img.get_data_dtype()}")
    print(f"  affine:")
    for row in img.affine:
        print(f"    [{row[0]:+.4f} {row[1]:+.4f} {row[2]:+.4f} {row[3]:+.2f}]")
    return img


print("=" * 70)
print("  ДИАГНОСТИКА AFFINE")
print("=" * 70)

qsm_img = header_info(QSM, "QSM")

for name in ["CrusCerebri_native", "SN_VTA_L_native", "SN_VTA_R_native"]:
    p = MASK_DIR / f"{name}.nii.gz"
    if p.exists():
        header_info(p, name)

# Проверка совпадения affine
print("\n" + "=" * 70)
print("  СРАВНЕНИЕ AFFINE")
print("=" * 70)

qsm_img = nib.load(str(QSM), mmap=False)
for name in ["CrusCerebri_native", "SN_VTA_L_native", "SN_VTA_R_native"]:
    p = MASK_DIR / f"{name}.nii.gz"
    if not p.exists():
        continue
    m = nib.load(str(p), mmap=False)
    same = np.allclose(qsm_img.affine, m.affine, atol=0.01)
    print(f"  {name}: affine {'✅ совпадает' if same else '❌ НЕ совпадает'}")

# Где находится маска Crus относительно QSM
print("\n" + "=" * 70)
print("  ГДЕ МАСКА CRUS В QSM")
print("=" * 70)

qsm = np.asarray(qsm_img.dataobj, dtype=np.float32)
crus_p = MASK_DIR / "CrusCerebri_native.nii.gz"

if crus_p.exists():
    crus = np.asarray(nib.load(str(crus_p)).dataobj) > 0.5
    print(f"  Маска Crus: {crus.sum()} вокселей")
    print(f"  QSM в маске Crus: min={qsm[crus].min():.4f}, "
          f"max={qsm[crus].max():.4f}, mean={qsm[crus].mean():.4f}")
    print(f"  Нулей в QSM внутри Crus: "
          f"{int((qsm[crus] == 0).sum())} из {crus.sum()} "
          f"({100*(qsm[crus] == 0).mean():.1f}%)")

    # Где маска в объёме
    coords = np.argwhere(crus)
    print(f"  Bounding box Crus:")
    print(f"    X: [{coords[:, 0].min()}, {coords[:, 0].max()}] "
          f"(QSM shape X = {qsm.shape[0]})")
    print(f"    Y: [{coords[:, 1].min()}, {coords[:, 1].max()}]")
    print(f"    Z: [{coords[:, 2].min()}, {coords[:, 2].max()}]")
    print(f"  Центр масс: {coords.mean(axis=0)}")

# Brain mask
if BRAIN_MASK.exists():
    bm = np.asarray(nib.load(str(BRAIN_MASK)).dataobj) > 0.5
    print(f"\n  Brain mask: {bm.sum()} вокселей")
    if crus_p.exists():
        overlap = (crus & bm).sum()
        print(f"  Crus ∩ brain_mask: {overlap} вокселей "
              f"({100*overlap/crus.sum():.1f}% Crus в мозге)")