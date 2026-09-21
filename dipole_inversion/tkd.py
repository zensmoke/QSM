"""Дипольная инверсия (Tikhonov / TKD) с zero-padding, float32."""
import numpy as np
from scipy.fft import fftn, ifftn


def dipole_kernel(shape, voxel_size):
    """
    Ядро диполя через broadcasting — без meshgrid.
    Экономит память в ~3 раза по сравнению с meshgrid+float64.
    """
    nx, ny, nz = shape
    kx = np.fft.fftfreq(nx, d=voxel_size[0]).astype(np.float32) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=voxel_size[1]).astype(np.float32) * 2 * np.pi
    kz = np.fft.fftfreq(nz, d=voxel_size[2]).astype(np.float32) * 2 * np.pi

    kx2 = kx ** 2
    ky2 = ky ** 2
    kz2 = kz ** 2

    # Broadcast: (nx,1,1) + (1,ny,1) + (1,1,nz)
    k2 = kx2[:, None, None] + ky2[None, :, None] + kz2[None, None, :]
    k2 = k2.astype(np.float32)
    k2[k2 == 0] = 1e-10

    D = (1.0 / 3.0 - kz2[None, None, :] / k2).astype(np.float32)
    D[0, 0, 0] = 0.0
    return D


def _padded_fft_inversion(local_field, voxel_size, lam=None, threshold=None,
                          pad_factor=1.5):
    """
    Общая функция: padding + инверсия.
    lam is None → TKD, иначе Tikhonov.
    """
    # --- Padding ---
    pads = [(0, int((pad_factor - 1) * s)) for s in local_field.shape]
    padded = np.pad(local_field, pads, mode="constant").astype(np.float32)
    crop = tuple(slice(0, s) for s in local_field.shape)

    D = dipole_kernel(padded.shape, voxel_size)
    field_k = fftn(padded)

    if lam is not None:
        # Tikhonov
        denom = (np.abs(D) ** 2 + lam).astype(np.float32)
        chi_k = np.conj(D).astype(np.complex64) * field_k.astype(np.complex64)
        chi_k /= denom
    else:
        # TKD
        D_max = float(np.abs(D).max())
        D_trunc = D.copy()
        small = np.abs(D) < threshold * D_max
        D_trunc[small] = np.sign(D[small] + 1e-12) * threshold * D_max
        D_trunc[np.abs(D_trunc) < 1e-8] = 1e-8
        chi_k = field_k / D_trunc

    chi_k[0, 0, 0] = 0.0                     # зануляем DC
    chi = np.real(ifftn(chi_k)).astype(np.float32)
    return chi[crop]


def _tkd(local_field, mask, voxel_size, threshold=0.15, **kwargs):
    threshold = float(threshold)
    print(f"[tkd] threshold={threshold}, padding=1.5x")
    chi = _padded_fft_inversion(local_field, voxel_size, lam=None,
                                 threshold=threshold, pad_factor=1.5)
    chi *= mask
    print(f"[tkd] χ диапазон: [{chi[mask].min():.4f}, {chi[mask].max():.4f}]")
    return chi


def _tikhonov(local_field, mask, voxel_size, lambda_=0.005, **kwargs):
    lam = float(kwargs.get("lambda", lambda_))
    print(f"[tikhonov] λ={lam}, padding=1.5x")
    chi = _padded_fft_inversion(local_field, voxel_size, lam=lam,
                                 threshold=None, pad_factor=1.5)
    chi *= mask
    print(f"[tikhonov] χ диапазон: [{chi[mask].min():.4f}, {chi[mask].max():.4f}]")
    return chi


def compute_qsm(local_field, mask, voxel_size=(1.0, 1.0, 1.0),
                method="tkd", **kwargs):
    if method == "tkd":
        return _tkd(local_field, mask, voxel_size, **kwargs)
    if method == "tikhonov":
        return _tikhonov(local_field, mask, voxel_size, **kwargs)
    raise ValueError(f"Неизвестный метод: {method}")