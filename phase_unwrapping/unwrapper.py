"""
Развёртка фазы:
  - 'laplacian' — Laplacian-based (Schofield & Zhu, 2003), чистый Python
  - 'romeo'     — ROMEO.jl через subprocess
  - 'prelude'   — FSL PRELUDE через subprocess
  - 'herraez'   — skimage.unwrap_phase (медленный)
  - 'none'      — без развёртки
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None


# ---------------------------------------------------------------------------
# Основная точка входа
# ---------------------------------------------------------------------------

def unwrap_phase_volume(phase, magnitude=None, mask=None,
                        method="laplacian", te=None, **kwargs):
    """
    Разворачивает фазу в 3D или 4D объёме.

    Args:
        phase:      (X,Y,Z) или (X,Y,Z,n_echo) массив фазы в радианах.
        magnitude:  (X,Y,Z) или (X,Y,Z,n_echo) — для ROMEO/PRELUDE.
        mask:       бинарная маска (опционально).
        method:     'laplacian' | 'romeo' | 'prelude' | 'herraez' | 'none'
        te:         времена эхо в мс (для ROMEO).

    Returns:
        Развёрнутая фаза той же формы, что и вход.
    """
    phase = np.asarray(phase, dtype=np.float32)

    if method == "none":
        print("[unwrap] Метод 'none' — фаза не разворачивается")
        return phase.copy()

    if method == "romeo":
        return _romeo_unwrap(phase, magnitude, mask, te)

    if method == "prelude":
        return _prelude_unwrap(phase, magnitude, mask)

    if method == "laplacian":
        if phase.ndim == 4:
            out = np.zeros_like(phase, dtype=np.float32)
            for e in range(phase.shape[-1]):
                print(f"[unwrap] Эхо {e + 1}/{phase.shape[-1]} (laplacian)")
                out[..., e] = _laplacian_unwrap(phase[..., e], mask)
            return out
        return _laplacian_unwrap(phase, mask)

    if method == "herraez":
        from skimage.restoration import unwrap_phase
        if phase.ndim == 4:
            out = np.zeros_like(phase, dtype=np.float32)
            for e in range(phase.shape[-1]):
                print(f"[unwrap] Эхо {e + 1}/{phase.shape[-1]} (herraez)")
                out[..., e] = _herraez_unwrap(phase[..., e], mask)
            return out
        return _herraez_unwrap(phase, mask)

    raise ValueError(f"Неизвестный метод развёртки: {method!r}")


# ---------------------------------------------------------------------------
# Laplacian unwrap (Schofield & Zhu, 2003)
# ---------------------------------------------------------------------------

def _laplacian_unwrap(phase, mask=None):
    from scipy.fft import fftn, ifftn, fftfreq

    if np.abs(phase).max() > np.pi * 1.001:
        phase = np.angle(np.exp(1j * phase.astype(np.float64))).astype(np.float32)

    nx, ny, nz = phase.shape
    kx = fftfreq(nx)[:, None, None]
    ky = fftfreq(ny)[None, :, None]
    kz = fftfreq(nz)[None, None, :]
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = 1.0

    psi = np.exp(1j * phase.astype(np.float64))
    gx = np.angle(psi * np.roll(np.conj(psi), -1, axis=0))
    gy = np.angle(psi * np.roll(np.conj(psi), -1, axis=1))
    gz = np.angle(psi * np.roll(np.conj(psi), -1, axis=2))

    dx = gx - np.roll(gx, 1, axis=0)
    dy = gy - np.roll(gy, 1, axis=1)
    dz = gz - np.roll(gz, 1, axis=2)
    rho = (dx + dy + dz).astype(np.float32)

    rho_k = fftn(rho)
    denom = -4 * np.pi ** 2 * k2
    phi_k = rho_k / denom
    phi_k[0, 0, 0] = 0.0
    unwrapped = np.real(ifftn(phi_k)).astype(np.float32)

    if mask is not None:
        m = mask.astype(bool)
        if m.sum() > 0:
            offset = np.median((phase - unwrapped)[m])
            unwrapped += offset
    else:
        offset = np.median(phase - unwrapped)
        unwrapped += offset

    return unwrapped.astype(np.float32)


# ---------------------------------------------------------------------------
# Herraez (skimage)
# ---------------------------------------------------------------------------

def _herraez_unwrap(phase, mask=None):
    from skimage.restoration import unwrap_phase

    if np.abs(phase).max() > np.pi * 1.001:
        phase = np.angle(np.exp(1j * phase.astype(np.float64))).astype(np.float32)

    p = phase * mask if mask is not None else phase
    try:
        out = unwrap_phase(p)
    except Exception as e:
        print(f"[unwrap] skimage упал: {e}, возвращаю фазу как есть")
        return phase.astype(np.float32)

    if mask is not None:
        out = out * mask
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# ROMEO через julia romeo.jl
# ---------------------------------------------------------------------------

def _romeo_unwrap(phase, magnitude, mask, te):
    if nib is None:
        raise RuntimeError("nibabel нужен для ROMEO-обёртки")

    phase4d = phase[..., None] if phase.ndim == 3 else phase
    mag4d = None
    if magnitude is not None:
        magnitude = np.asarray(magnitude, dtype=np.float32)
        mag4d = magnitude[..., None] if magnitude.ndim == 3 else magnitude

    with tempfile.TemporaryDirectory(prefix="romeo_") as tmpdir:
        tmp = Path(tmpdir)
        phase_path = tmp / "phase.nii.gz"
        mag_path = tmp / "mag.nii.gz" if mag4d is not None else None

        nib.save(nib.Nifti1Image(phase4d.astype(np.float32), np.eye(4)),
                 str(phase_path))
        if mag_path is not None:
            nib.save(nib.Nifti1Image(mag4d.astype(np.float32), np.eye(4)),
                     str(mag_path))

        from phase_unwrapping.romeo import unwrap_with_romeo
        out_path = unwrap_with_romeo(phase_path, tmp, mag_path, te)

        unwrapped = np.asarray(nib.load(str(out_path)).dataobj,
                                dtype=np.float32)

    if phase.ndim == 3 and unwrapped.ndim == 4:
        unwrapped = unwrapped[..., 0]
    if phase.ndim == 4 and unwrapped.ndim == 3:
        unwrapped = unwrapped[..., None]

    while unwrapped.ndim > phase.ndim:
        unwrapped = unwrapped[..., 0]

    if mask is not None:
        if unwrapped.ndim == 4:
            unwrapped = unwrapped * mask[..., None]
        else:
            unwrapped = unwrapped * mask

    return unwrapped.astype(np.float32)


# ---------------------------------------------------------------------------
# PRELUDE через FSL
# ---------------------------------------------------------------------------

def _prelude_unwrap(phase, magnitude, mask):
    """
    Обёртка FSL PRELUDE.
    ВАЖНО: PRELUDE требует оба файла: phase (-p) и abs (-a, магнитуда).
    """
    if nib is None:
        raise RuntimeError("nibabel нужен для PRELUDE-обёртки")
    if magnitude is None:
        raise ValueError(
            "PRELUDE требует магнитуду (abs). Передайте magnitude в "
            "unwrap_phase_volume или выберите другой метод."
        )

    # 3D → 4D (PRELUDE работает с 4D, обрабатывая каждое эхо)
    phase4d = phase[..., None] if phase.ndim == 3 else phase
    mag4d = magnitude[..., None] if magnitude.ndim == 3 else magnitude

    with tempfile.TemporaryDirectory(prefix="prelude_") as tmpdir:
        tmp = Path(tmpdir)

        phase_path = tmp / "phase.nii.gz"
        mag_path = tmp / "mag.nii.gz"
        out_base = tmp / "unwrapped"

        nib.save(nib.Nifti1Image(phase4d.astype(np.float32), np.eye(4)),
                 str(phase_path))
        nib.save(nib.Nifti1Image(mag4d.astype(np.float32), np.eye(4)),
                 str(mag_path))

        # Опциональная маска
        safe_mask = None
        if mask is not None:
            safe_mask = tmp / "mask.nii.gz"
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)),
                     str(safe_mask))

        from phase_unwrapping.prelude import run_prelude
        out_path = run_prelude(
            phase_path, mag_path, out_base,
            mask_path=safe_mask,
        )

        unwrapped = np.asarray(nib.load(str(out_path)).dataobj,
                                dtype=np.float32)

    # Приведение формы
    if phase.ndim == 3 and unwrapped.ndim == 4:
        unwrapped = unwrapped[..., 0]
    if phase.ndim == 4 and unwrapped.ndim == 3:
        unwrapped = unwrapped[..., None]

    while unwrapped.ndim > phase.ndim:
        unwrapped = unwrapped[..., 0]

    return unwrapped.astype(np.float32)


# ---------------------------------------------------------------------------
# Объединение эхо (взвешенное по TE)
# ---------------------------------------------------------------------------

def combine_echoes(phase, te, mask=None):
    phase = np.asarray(phase, dtype=np.float32)
    if phase.ndim == 3:
        return phase

    te = np.asarray(te, dtype=np.float32)
    if np.all(te == 0):
        weights = np.ones(phase.shape[-1], dtype=np.float32)
    else:
        weights = te / te.sum()

    combined = np.zeros(phase.shape[:3], dtype=np.float32)
    for e in range(phase.shape[-1]):
        combined += weights[e] * phase[..., e]

    return combined