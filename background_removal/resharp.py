"""
RESHARP (Regularization Enabled SHARP) для удаления фонового поля.
"""
import numpy as np
from scipy import ndimage
from scipy.fft import fftn, ifftn


def _create_spherical_kernel(radius_mm, voxel_size):
    """Создает нормированное сферическое ядро."""
    voxel_size_mean = float(np.mean(voxel_size))
    radius_vox = radius_mm / voxel_size_mean

    size = int(2 * np.ceil(radius_vox) + 1)
    center = size // 2
    z, y, x = np.ogrid[-center:size - center,
                       -center:size - center,
                       -center:size - center]
    sphere = (x**2 + y**2 + z**2) <= radius_vox**2
    kernel = sphere.astype(np.float32)
    kernel /= kernel.sum()
    return kernel


def remove_background_resharp(phase, mask,
                               voxel_size=(1.0, 1.0, 1.0),
                               lambda_reg=0.01,
                               radius_mm=5.0,
                               erode_iter=0,
                               verbose=True):
    """Удаление фонового поля через RESHARP."""
    if verbose:
        print(f"[RESHARP] λ={lambda_reg}, radius={radius_mm}mm, "
              f"erode_iter={erode_iter}")

    phase = np.asarray(phase, dtype=np.float32)
    mask = mask.astype(bool)

    # Эрозия
    if erode_iter > 0:
        mask_work = ndimage.binary_erosion(mask, iterations=erode_iter)
        if verbose:
            print(f"[RESHARP] Эрозия: {erode_iter} итер., "
                  f"{int(mask_work.sum())} вокселей")
    else:
        mask_work = mask

    # Ядро
    kernel = _create_spherical_kernel(radius_mm, voxel_size)
    if verbose:
        print(f"[RESHARP] Размер ядра: {kernel.shape}")

    # SMV-свертка для получения оценки фона
    bg_field_smv = ndimage.convolve(phase * mask_work, kernel,
                                     mode='constant', cval=0.0)

    # Оператор (δ - ρ)
    delta_kernel = np.zeros_like(kernel)
    center = tuple(s // 2 for s in kernel.shape)
    delta_kernel[center] = 1.0
    operator_kernel = delta_kernel - kernel

    # Переносим в k-пространство
    operator_k = fftn(operator_kernel, s=phase.shape)
    phase_k = fftn(phase)

    # Решение Тихонова: B_local = conj(K) * K * B_total / (|K|² + λ)
    numerator = np.conj(operator_k) * (operator_k * phase_k)
    denominator = np.abs(operator_k) ** 2 + lambda_reg
    local_field_k = numerator / denominator

    # Обратное FFT
    local_field = np.real(ifftn(local_field_k)).astype(np.float32)
    local_field *= mask_work

    if verbose:
        m = mask_work > 0
        print(f"[RESHARP] Диапазон локального поля: "
              f"[{local_field[m].min():.4f}, {local_field[m].max():.4f}], "
              f"std={local_field[m].std():.4f}")

    return local_field