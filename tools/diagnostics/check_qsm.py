import nibabel as nib
import numpy as np

'''
qsm = nib.load("/Users/nikitamyasov/3782185 ласточка/output/qsm.nii.gz").get_fdata()
mask = nib.load("/Users/nikitamyasov/3782185 ласточка/output/mask.nii.gz").get_fdata() > 0.5

print(f"qsm global:  min={qsm.min():.6f}, max={qsm.max():.6f}, "
      f"std={qsm.std():.6f}")
print(f"qsm в маске: min={qsm[mask].min():.6f}, max={qsm[mask].max():.6f}, "
      f"std={qsm[mask].std():.6f}, mean={qsm[mask].mean():.6f}")
print(f"Перцентили внутри маски:")
for p in (1, 5, 25, 50, 75, 95, 99):
    print(f"   {p:>3}%: {np.percentile(qsm[mask], p):.6f}")
'''
'''
import nibabel as nib
import numpy as np

mask = nib.load('/Users/nikitamyasov/3782185 ласточка/output/mask.nii.gz').get_fdata() > 0.5
mag  = nib.load('/Users/nikitamyasov/3782185 ласточка/data/echo-1_part-mag.nii').get_fdata()

print(f"Объём: {mask.size} вокселей")
print(f"Маска: {mask.sum()} вокселей ({100*mask.mean():.2f}%)")
print(f"Магнитуда внутри маски: "
      f"min={mag[mask].min():.2f}, "
      f"median={np.median(mag[mask]):.2f}, "
      f"max={mag[mask].max():.2f}")
print(f"Магнитуда ВНЕ маски:   "
      f"median={np.median(mag[~mask]):.2f}")
'''

'''
import nibabel as nib
import numpy as np

qsm = nib.load("/Users/nikitamyasov/3782185 ласточка/output/qsm.nii.gz").get_fdata()
mask = nib.load("/Users/nikitamyasov/3782185 ласточка/output/brain_mask.nii.gz").get_fdata() > 0.5

print(f"Глобально: min={qsm.min():.4f}, max={qsm.max():.4f}, "
      f"std={qsm.std():.4f}")
print(f"В маске:   min={qsm[mask].min():.4f}, max={qsm[mask].max():.4f}, "
      f"std={qsm[mask].std():.4f}, mean={qsm[mask].mean():.4f}")
print()
print("Перцентили внутри маски:")
for p in (1, 5, 25, 50, 75, 95, 99):
    print(f"  {p:>3}%: {np.percentile(qsm[mask], p):+.4f}")

# Проверим специфичные области (базальные ганглии примерно по центру)
nx, ny, nz = qsm.shape
print(f"\nЗначения в центре среза z={nz//2}:")
print(f"  Центр ({nx//2},{ny//2}): {qsm[nx//2, ny//2, nz//2]:+.4f}")
'''

'''
import nibabel as nib
import numpy as np

lf = nib.load("/Users/nikitamyasov/3782185 ласточка/output/local_field.nii.gz").get_fdata()
mask = nib.load("/Users/nikitamyasov/3782185 ласточка/output/brain_mask.nii.gz").get_fdata() > 0.5
pc = nib.load("/Users/nikitamyasov/3782185 ласточка/output/phase_combined.nii.gz").get_fdata()

print(f"phase_combined: min={pc.min():.4f}, max={pc.max():.4f}, std={pc.std():.4f}")
print(f"local_field (в маске): min={lf[mask].min():.4e}, "
      f"max={lf[mask].max():.4e}, std={lf[mask].std():.4e}")
'''

import nibabel as nib
import numpy as np

# Открыть unwrapped_phase
u = nib.load("/Users/nikitamyasov/3782185 ласточка/output/unwrapped_phase.nii.gz").get_fdata()
mask = nib.load("/Users/nikitamyasov/3782185 ласточка/output/mask.nii.gz").get_fdata() > 0.5

print(f"Форма: {u.shape}")
print(f"dtype: {u.dtype}")
print(f"Global: min={u.min():.3f}, max={u.max():.3f}, std={u.std():.3f}")

# Эхо 1
u1 = u[..., 0]
print(f"\nЭхо 1: min={u1.min():.3f}, max={u1.max():.3f}, std={u1.std():.3f}")
print(f"Эхо 1 в маске: std={u1[mask].std():.3f}, mean={u1[mask].mean():.3f}")

# Сколько уникальных значений?
n_unique = len(np.unique(np.round(u1, 3)))
print(f"Уникальных значений (эхо 1): {n_unique}")
