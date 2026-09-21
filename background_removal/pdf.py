"""Projection onto Dipole Fields (PDF) с демпфированием."""
import numpy as np
from scipy import ndimage
from scipy.fft import fftn, ifftn, fftfreq


def dipole_kernel(shape, voxel_size):
    nx, ny, nz = shape
    kx = fftfreq(nx, d=voxel_size[0]) * 2 * np.pi
    ky = fftfreq(ny, d=voxel_size[1]) * 2 * np.pi
    kz = fftfreq(nz, d=voxel_size[2]) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    k2[k2 == 0] = 1e-10
    D = (1.0 / 3.0 - KZ ** 2 / k2).astype(np.float64)
    D[0, 0, 0] = 0.0
    return D


def remove_background_pdf(phase, mask, voxel_size=(1.0, 1.0, 1.0),
                          tol=1e-3, max_iter=50, erode_iter=3,
                          reg=0.05, alpha=0.5, verbose=True):
    """
    PDF с демпфированием (alpha).

    alpha — коэффициент релаксации (0..1):
        1.0 = полное обновление phi_bg (может расходиться)
        0.5 = половинное (рекомендуется)
        0.2 = очень осторожное
    """
    tol = float(tol)
    reg = float(reg)
    max_iter = int(max_iter)
    erode_iter = int(erode_iter)
    alpha = float(alpha)
    voxel_size = tuple(float(v) for v in voxel_size)

    phase = np.asarray(phase, dtype=np.float64)
    mask = mask.astype(bool)

    # --- Эрозия маски ---
    if erode_iter > 0:
        mask_work = ndimage.binary_erosion(mask, iterations=erode_iter)
        if verbose:
            print(f"[PDF] Эрозия: {erode_iter} итер., {int(mask_work.sum())} "
                  f"вокселей (было {int(mask.sum())})")
    else:
        mask_work = mask

    if mask_work.sum() < 50:
        raise ValueError("Маска слишком мала после эрозии")

    mask_f = mask_work.astype(np.float64)
    inv_mask = 1.0 - mask_f

    D = dipole_kernel(phase.shape, voxel_size)
    abs_D2 = np.abs(D) ** 2

    if verbose:
        print(f"[PDF] tol={tol}, max_iter={max_iter}, reg={reg}, alpha={alpha}")

    phi_bg = np.zeros_like(phase)
    prev_delta = None

    for it in range(max_iter):
        # Остаток
        residual = (phase - phi_bg) * inv_mask

        # Инверсия с Тихоновской регуляризацией
        residual_k = fftn(residual)
        chi_k = np.conj(D) * residual_k / (abs_D2 + reg)
        chi = np.real(ifftn(chi_k)) * inv_mask

        # Прямая задача
        new_phi_bg = np.real(ifftn(D * fftn(chi)))

        # --- ДЕМПФИРОВАНИЕ ---
        new_phi_bg = alpha * new_phi_bg + (1 - alpha) * phi_bg

        delta = float(np.abs(new_phi_bg - phi_bg).max())
        phi_bg = new_phi_bg

        if verbose and (it < 5 or it % 10 == 0 or delta < tol):
            print(f"[PDF] Итерация {it + 1:3d}: delta = {delta:.4e}")

        if delta < tol:
            if verbose:
                print(f"[PDF] Сошлось на итерации {it + 1}")
            break

        # Проверка на взрыв (порог выше, т.к. фаза у вас большая)
        if np.abs(phi_bg).max() > 100:
            if verbose:
                print(f"[PDF] ⚠ phi_bg max = {np.abs(phi_bg).max():.2e} "
                      f"на итерации {it + 1}, останавливаюсь")
            break

        if prev_delta is not None and delta > prev_delta * 5 and it > 2:
            if verbose:
                print(f"[PDF] ⚠ delta растёт ({prev_delta:.3e} → {delta:.3e})")
            break
        prev_delta = delta

    local_field = (phase - phi_bg) * mask_f

    if verbose:
        m = mask_f > 0
        print(f"[PDF] Диапазон локального поля: "
              f"[{local_field[m].min():.4f}, {local_field[m].max():.4f}], "
              f"std={local_field[m].std():.4f}")

    return local_field.astype(np.float32)


def remove_background(phase, mask, voxel_size=(1.0, 1.0, 1.0),
                      method="pdf", **kwargs):
    """
    Универсальный диспетчер для удаления фонового поля.

    Поддерживает:
      - 'pdf'      → Projection onto Dipole Fields
      - 'resharp'  → Regularization Enabled SHARP
      - 'none'     → без удаления
    """
    if method in ("none", None, ""):
        return phase.copy()

    if method == "pdf":
        return remove_background_pdf(phase, mask, voxel_size, **kwargs)

    if method == "resharp":
        # Ленивый импорт — избегаем циклической зависимости
        from background_removal.resharp import remove_background_resharp
        return remove_background_resharp(
            phase, mask, voxel_size=voxel_size, **kwargs,
        )

    raise ValueError(f"Неизвестный метод: {method!r}")