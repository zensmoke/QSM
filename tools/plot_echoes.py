"""
Визуализация всех эхо из unwrapped_phase.nii.gz.

Делает одно изображение с сеткой subplot'ов: по одному срезу на эхо.
Полезно для визуальной проверки качества развёртки фазы.
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_echoes(nifti_path: Path,
                output_png: Path | None = None,
                slice_axis: int = 2,
                slice_index: int | None = None,
                cmap: str = "gray",
                per_echo_scale: bool = True,
                n_cols: int = 4,
                figsize_per_panel: tuple = (3.5, 3.5),
                dpi: int = 120,
                mask_path: Path | None = None) -> None:
    """
    Строит сетку subplot'ов — по одному срезу на каждое эхо.

    Args:
        nifti_path: путь к 4D NIfTI (unwrapped_phase.nii.gz).
        output_png: путь для сохранения (если None — только показать).
        slice_axis: 0 | 1 | 2 — по какой оси брать срез (2 = аксиальный).
        slice_index: номер среза. None → середина объёма.
        cmap: цветовая карта ('gray', 'viridis', 'RdBu_r', 'jet').
        per_echo_scale: если True — каждое эхо в своём диапазоне.
                        Если False — общая шкала для всех.
        n_cols: сколько колонок в сетке.
        figsize_per_panel: размер одной панели в дюймах.
        dpi: разрешение.
        mask_path: опциональная маска — вне маски будет NaN.
    """
    # --- Загрузка ---
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    print(f"Форма: {data.shape}")
    print(f"dtype: {data.dtype}")

    # Приводим к 4D
    if data.ndim == 3:
        data = data[..., None]
        print("[info] 3D → 4D (одно эхо)")

    n_echoes = data.shape[-1]
    print(f"Число эхо: {n_echoes}")

    # --- Опциональная маска ---
    mask = None
    if mask_path is not None and Path(mask_path).exists():
        mask_data = np.asarray(nib.load(str(mask_path)).dataobj)
        mask = mask_data > 0.5
        print(f"Маска: {mask.sum()} вокселей")

    # --- Выбор среза ---
    n_slices = data.shape[slice_axis]
    if slice_index is None:
        slice_index = n_slices // 2
    slice_index = int(np.clip(slice_index, 0, n_slices - 1))
    print(f"Ось среза: {slice_axis}, индекс: {slice_index} / {n_slices - 1}")

    # --- Функция для извлечения среза ---
    def get_slice(volume_3d):
        if slice_axis == 0:
            return volume_3d[slice_index, :, :].T
        elif slice_axis == 1:
            return volume_3d[:, slice_index, :].T
        else:  # slice_axis == 2
            return volume_3d[:, :, slice_index].T

    def get_mask_slice(mask_3d):
        if slice_axis == 0:
            return mask_3d[slice_index, :, :].T
        elif slice_axis == 1:
            return mask_3d[:, slice_index, :].T
        else:
            return mask_3d[:, :, slice_index].T

    # --- Сетка subplot'ов ---
    n_rows = int(np.ceil(n_echoes / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols,
                 figsize_per_panel[1] * n_rows),
        dpi=dpi,
    )
    axes = np.atleast_1d(axes).ravel()

    # --- Общая шкала (если per_echo_scale=False) ---
    if not per_echo_scale:
        all_slices = [get_slice(data[..., e]) for e in range(n_echoes)]
        if mask is not None:
            mask_slice = get_mask_slice(mask)
            all_vals = np.concatenate([s[mask_slice].ravel()
                                       for s in all_slices])
        else:
            all_vals = np.concatenate([s.ravel() for s in all_slices])
        global_min = float(np.percentile(all_vals, 1))
        global_max = float(np.percentile(all_vals, 99))
        print(f"Общая шкала: [{global_min:.3f}, {global_max:.3f}]")

    # --- Рисуем каждое эхо ---
    for e in range(n_echoes):
        ax = axes[e]
        sl = get_slice(data[..., e]).astype(np.float32)

        # Маскирование
        if mask is not None:
            m_sl = get_mask_slice(mask)
            sl = np.where(m_sl, sl, np.nan)

        # Шкала
        if per_echo_scale:
            finite = sl[np.isfinite(sl)]
            vmin = float(np.percentile(finite, 1))
            vmax = float(np.percentile(finite, 99))
        else:
            vmin, vmax = global_min, global_max

        im = ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax,
                        origin="lower", interpolation="nearest")
        ax.set_title(f"Эхо {e + 1}/{n_echoes}", fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Скрываем пустые панели
    for i in range(n_echoes, len(axes)):
        axes[i].axis("off")

    # --- Общий заголовок ---
    fig.suptitle(
        f"unwrapped_phase.nii.gz  ·  срез {slice_index}/{n_slices - 1} "
        f"(ось {slice_axis})\n"
        f"шкала: {'индивидуальная' if per_echo_scale else 'общая'}",
        fontsize=13, y=1.00,
    )

    plt.tight_layout()

    # --- Сохранение ---
    if output_png is not None:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_png), dpi=dpi, bbox_inches="tight")
        print(f"Сохранено: {output_png}")

    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Визуализация всех эхо из unwrapped_phase.nii.gz"
    )
    parser.add_argument(
        "nifti_path", nargs="?",
        default="/Users/nikitamyasov/3782185 ласточка/output/unwrapped_phase.nii.gz",
        help="Путь к unwrapped_phase.nii.gz",
    )
    parser.add_argument(
        "-o", "--output",
        default="/Users/nikitamyasov/3782185 ласточка/output/echoes_grid.png",
        help="Куда сохранить PNG",
    )
    parser.add_argument(
        "--slice-axis", type=int, default=2, choices=[0, 1, 2],
        help="Ось среза: 0=sagittal, 1=coronal, 2=axial (default)",
    )
    parser.add_argument(
        "--slice-index", type=int, default=None,
        help="Номер среза (по умолчанию середина)",
    )
    parser.add_argument(
        "--cmap", default="gray",
        help="Colormap: gray, viridis, RdBu_r, jet (default: gray)",
    )
    parser.add_argument(
        "--common-scale", action="store_true",
        help="Использовать общую шкалу для всех эхо",
    )
    parser.add_argument(
        "--mask", default=None,
        help="Опциональная маска (обнулит фон)",
    )
    parser.add_argument(
        "--cols", type=int, default=4,
        help="Число колонок в сетке (default: 4)",
    )
    args = parser.parse_args()

    plot_echoes(
        nifti_path=Path(args.nifti_path),
        output_png=Path(args.output) if args.output else None,
        slice_axis=args.slice_axis,
        slice_index=args.slice_index,
        cmap=args.cmap,
        per_echo_scale=not args.common_scale,
        n_cols=args.cols,
        mask_path=Path(args.mask) if args.mask else None,
    )


if __name__ == "__main__":
    main()