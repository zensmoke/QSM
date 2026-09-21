"""Загрузка QSM-данных из папки или явного списка файлов."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import nibabel as nib

from utils.nifti_utils import load_nifti
from preprocessing.rename_files import read_te_from_json


class QSMData:
    def __init__(self):
        self.magnitude: np.ndarray | None = None
        self.phase: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.affine: np.ndarray | None = None
        self.header: nib.Nifti1Header | None = None
        self.te: list[float] = []
        self.voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0)


def _stack_echoes(paths: list[Path]):
    vols = []
    affine = header = None
    for i, p in enumerate(paths):
        data, aff, hdr = load_nifti(p)
        vols.append(data)
        if i == 0:
            affine, header = aff, hdr
    return np.stack(vols, axis=-1), affine, header


def _te_from_json_list(json_paths: list[Path]) -> list[float]:
    tes = []
    for jp in json_paths:
        te = read_te_from_json(jp) if jp and jp.exists() else None
        tes.append(te if te is not None else 0.0)
    print(f"[data] TE из JSON: {tes}")
    return tes


def load_qsm_data(mag_paths: list[Path],
                  phase_paths: list[Path],
                  te: list[float] | None = None,
                  json_mag_paths: list[Path] | None = None,
                  mask_path: Path | None = None) -> QSMData:
    """Загружает QSM-данные (магнитуда, фаза, маска, метаданные)."""
    assert len(mag_paths) == len(phase_paths), \
        f"Число mag ({len(mag_paths)}) и phase ({len(phase_paths)}) не совпадает"

    qsm = QSMData()
    qsm.magnitude, qsm.affine, qsm.header = _stack_echoes(mag_paths)
    qsm.phase, _, _ = _stack_echoes(phase_paths)

    # Приоритет TE: явный аргумент > JSON > нули
    if te and any(t > 0 for t in te):
        qsm.te = list(te)
        print(f"[data] TE (из config): {qsm.te}")
    elif json_mag_paths:
        qsm.te = _te_from_json_list(json_mag_paths)
    else:
        qsm.te = [0.0] * len(mag_paths)

    if qsm.header is not None:
        zooms = qsm.header.get_zooms()
        qsm.voxel_size = tuple(float(z) for z in zooms[:3])

    if mask_path is not None:
        qsm.mask, _, _ = load_nifti(mask_path)
        qsm.mask = qsm.mask > 0.5
    else:
        qsm.mask = _auto_mask(qsm.magnitude)

    print(f"[data] Магнитуда: {qsm.magnitude.shape}")
    print(f"[data] Фаза:      {qsm.phase.shape}")
    print(f"[data] Маска:     {qsm.mask.shape}, вокселей: {int(qsm.mask.sum())}")
    print(f"[data] Voxel size: {qsm.voxel_size}")
    print(f"[data] TE (сек):  {qsm.te}")

    return qsm


def _auto_mask(magnitude: np.ndarray) -> np.ndarray:
    """
    Авто-маска для QSM с магнитудой хорошего контраста.
    Использует Otsu как жёсткий порог + largest component + fill holes.
    """
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # Усредняем по эхо — стабильнее, чем брать только первое
    mag_mean = magnitude.mean(axis=-1) if magnitude.ndim == 4 else magnitude

    # Otsu — автоматический порог между "ткань" и "фон"
    thr_otsu = float(threshold_otsu(mag_mean))

    # ЖЁСТКИЙ порог: только Otsu без понижающего множителя
    # Если Otsu даёт слишком много — можно поднять до 1.2 * thr_otsu
    mask = mag_mean > thr_otsu

    # Морфологическая очистка
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)

    # Оставляем только крупные связные компоненты (>10% от самой большой)
    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        biggest = sizes.max()
        keep = [i + 1 for i, s in enumerate(sizes) if s > 0.1 * biggest]
        mask = np.isin(labels, keep)

    print(f"[auto_mask] Otsu={thr_otsu:.1f}, "
          f"маска = {100 * mask.mean():.2f}% объёма, "
          f"{int(mask.sum())} вокселей")

    return mask