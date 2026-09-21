"""
Главный скрипт QSM-пайплайна.

Все пути ОБЯЗАТЕЛЬНО указываются через CLI:
  --data-dir      папка с NIfTI (echo-*_part-mag/phase, JSON)
  --output-dir    папка для результатов QSM
  --mask-path     готовая маска мозга (опционально)
  --config        путь к config.yaml (по умолчанию: ./config.yaml)

Полный цикл:
  1. Переименование файлов in-place
  2. Загрузка данных + построение маски (FSL BET / HD-BET / Otsu)
  3. Развёртка фазы (laplacian / romeo / prelude)
  4. Объединение эхо
  5. Удаление фонового поля (PDF / RESHARP)
  6. Дипольная инверсия (Tikhonov / TKD) + ppm + центрирование
  7. Сохранение результатов

Пример запуска:
    python qsm.py \\
        --data-dir /путь/к/data \\
        --output-dir /путь/к/output
"""
import argparse
import sys
import re
from pathlib import Path

import yaml
import numpy as np

from preprocessing.rename_files import rename_qsm_files
from qsm_io.data_loader import load_qsm_data
from phase_unwrapping.unwrapper import unwrap_phase_volume, combine_echoes
from background_removal.pdf import remove_background
from dipole_inversion.tkd import compute_qsm
from utils.nifti_utils import save_nifti


GAMMA_PROTON = 2 * np.pi * 42.57747892e6   # рад/(с·Тл)


# =============================================================================
# УТИЛИТЫ
# =============================================================================

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sorted_by_echo(paths):
    def key(p):
        m = re.search(r'echo-(\d+)', p.name)
        return int(m.group(1)) if m else 0
    return sorted(paths, key=key)


def _diagnose(tag, arr, mask=None):
    print(f"[DIAG {tag}] global: min={arr.min():.4f}, max={arr.max():.4f}, "
          f"std={arr.std():.4f}")
    if mask is not None:
        m = mask.astype(bool)
        if m.sum() > 0:
            print(f"[DIAG {tag}] в маске: min={arr[m].min():.4f}, "
                  f"max={arr[m].max():.4f}, std={arr[m].std():.4f}, "
                  f"mean={arr[m].mean():.4f}")


def _normalize_phase(phase, label=""):
    """Приводит фазу к радианам в [-π, π]."""
    pmax = float(np.abs(phase).max())
    tag = f"[phase{(' ' + label) if label else ''}]"
    print(f"{tag} вход: min={phase.min():.3f}, max={phase.max():.3f}, "
          f"std={phase.std():.3f}")

    if pmax <= np.pi * 1.001:
        print(f"{tag} → радианы в [-π, π]")
        return phase.astype(np.float32)

    if 100 < pmax < np.pi * 1000 * 1.001:
        print(f"{tag} → миллирадианы (max|φ|={pmax:.0f}), делю на 1000")
        phase_rad = (phase / 1000.0).astype(np.float32)
        print(f"{tag} после /1000: min={phase_rad.min():.3f}, "
              f"max={phase_rad.max():.3f}, std={phase_rad.std():.3f}")
        return phase_rad

    if 3500 < pmax < 5000 and phase.min() >= 0:
        print(f"{tag} → DICOM 12-bit")
        phase_rad = (phase / 4096.0 * 2 * np.pi - np.pi).astype(np.float32)
        return phase_rad

    print(f"{tag} → wrap через exp(i·φ)")
    phase_rad = np.angle(np.exp(1j * phase.astype(np.float64))).astype(np.float32)
    print(f"{tag} после wrap: min={phase_rad.min():.3f}, "
          f"max={phase_rad.max():.3f}, std={phase_rad.std():.3f}")
    return phase_rad


# =============================================================================
# ПАЙПЛАЙН
# =============================================================================

def run_pipeline(cfg: dict, data_dir: Path, output_dir: Path,
                  mask_path_override: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        print(f"[FATAL] Папка не найдена: {data_dir}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  QSM PIPELINE")
    print("=" * 65)
    print(f"  data_dir:   {data_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  mask:       {mask_path_override or 'авто'}")

    # =====================================================================
    # ШАГ 1. Переименование
    # =====================================================================
    if cfg["preprocessing"].get("rename_files", True):
        print("\n=== ШАГ 1: Переименование (in-place) ===")
        rename_qsm_files(
            data_dir,
            series_filter=cfg["preprocessing"].get("series_filter"),
            dry_run=cfg["preprocessing"].get("dry_run", False),
        )
        if cfg["preprocessing"].get("dry_run", False):
            print("[main] DRY-RUN активен, выходим")
            return
    else:
        print("\n=== ШАГ 1: Пропущено ===")

    mag_paths = _sorted_by_echo([
        p for p in data_dir.glob("echo-*_part-mag.nii*")
        if not p.name.endswith(".json")
    ])
    phase_paths = _sorted_by_echo([
        p for p in data_dir.glob("echo-*_part-phase.nii*")
        if not p.name.endswith(".json")
    ])
    json_mag_paths = _sorted_by_echo(list(data_dir.glob("echo-*_part-mag.json")))

    print(f"[main] mag={len(mag_paths)}, phase={len(phase_paths)}, "
          f"json_mag={len(json_mag_paths)}")

    if not mag_paths or not phase_paths:
        print("[FATAL] Не найдено пар mag/phase в", data_dir)
        sys.exit(1)

    # =====================================================================
    # ШАГ 2. Маска + загрузка
    # =====================================================================
    print("\n=== ШАГ 2: Загрузка данных ===")

    mask_cfg = cfg["input"].get("mask_path")
    mask_source = cfg["input"].get("mask_source", "auto")
    mask_path = None

    if mask_path_override and mask_path_override.exists():
        mask_path = mask_path_override
        print(f"[main] Маска из CLI: {mask_path}")
    elif mask_cfg and Path(mask_cfg).expanduser().exists():
        mask_path = Path(mask_cfg).expanduser()
        print(f"[main] Маска из config: {mask_path}")
    elif (output_dir / "brain_mask.nii.gz").exists() and mask_source != "otsu":
        mask_path = output_dir / "brain_mask.nii.gz"
        print(f"[main] Маска найдена: {mask_path}")
    else:
        if mask_source in ("fsl", "auto"):
            try:
                from qsm_io.fsl_bet import run_fsl_bet
                print("[main] Запускаю FSL BET...")
                mask_path = run_fsl_bet(
                    mag_paths[0], output_dir,
                    fractional_intensity=cfg["input"].get("bet_f", 0.4),
                    robust=cfg["input"].get("bet_robust", True),
                )
            except Exception as e:
                print(f"[main] FSL BET не удался: {e}")
                if mask_source == "fsl":
                    raise
                mask_path = None
        elif mask_source == "hdbet":
            try:
                from qsm_io.hdbet import run_hdbet
                print("[main] Запускаю HD-BET...")
                mask_path = run_hdbet(mag_paths[0], output_dir)
            except Exception as e:
                print(f"[main] HD-BET не удался: {e}")
                mask_path = None

        if mask_path is None:
            print("[main] Маска не создана — авто-Otsu")

    acq = cfg.get("acquisition", {})
    qsm = load_qsm_data(
        mag_paths, phase_paths,
        te=acq.get("TE") or None,
        json_mag_paths=json_mag_paths,
        mask_path=mask_path,
    )

    if acq.get("voxel_size"):
        qsm.voxel_size = tuple(acq["voxel_size"])

    mask_frac = qsm.mask.mean()
    print(f"[main] Маска: {100 * mask_frac:.1f}% объёма "
          f"({int(qsm.mask.sum())} вокселей)")

    qsm.phase = _normalize_phase(qsm.phase, label="[4D]")

    # =====================================================================
    # ШАГ 3. Развёртка фазы
    # =====================================================================
    print("\n=== ШАГ 3: Развёртка фазы ===")
    unwrap_method = cfg["phase_unwrapping"]["method"]
    print(f"[main] Метод: {unwrap_method}")

    unwrap_kwargs = {"mask": qsm.mask, "method": unwrap_method}
    if unwrap_method in ("romeo", "laplacian"):
        unwrap_kwargs["magnitude"] = qsm.magnitude
    if unwrap_method == "romeo":
        unwrap_kwargs["te"] = [t * 1000.0 for t in qsm.te] if qsm.te else None

    try:
        unwrapped = unwrap_phase_volume(qsm.phase, **unwrap_kwargs)
    except TypeError as e:
        print(f"[main] Fallback ({e})")
        unwrapped = unwrap_phase_volume(
            qsm.phase, mask=qsm.mask, method=unwrap_method,
        )

    save_nifti(unwrapped, qsm.affine, qsm.header,
               output_dir / "unwrapped_phase.nii.gz")
    _diagnose("unwrapped", unwrapped, qsm.mask)

    # =====================================================================
    # ШАГ 4. Объединение эхо
    # =====================================================================
    print("\n=== ШАГ 4: Объединение эхо ===")
    proc = cfg["processing"]
    if proc["echo_strategy"] == "combined":
        phase_combined = combine_echoes(unwrapped, qsm.te, qsm.mask)
    else:
        idx = proc["single_echo_index"]
        phase_combined = unwrapped[..., idx] * qsm.mask
        print(f"[combine] Эхо #{idx}")

    mean_phase = float(phase_combined[qsm.mask].mean())
    print(f"[main] DC фазы: {mean_phase:+.4f} рад")
    phase_combined = phase_combined - mean_phase

    save_nifti(phase_combined, qsm.affine, qsm.header,
               output_dir / "phase_combined.nii.gz")
    _diagnose("phase_combined", phase_combined, qsm.mask)

    # =====================================================================
    # ШАГ 5. Удаление фонового поля
    # =====================================================================
    bg_cfg = cfg["background_removal"]
    if not bg_cfg.get("enabled", True) or bg_cfg.get("method") == "none":
        print("\n=== ШАГ 5: BFR ПРОПУЩЕН ===")
        local_field = phase_combined.copy()
    else:
        method = bg_cfg.get("method", "pdf")
        print(f"\n=== ШАГ 5: BFR ({method}) ===")
        bg_kwargs = dict(bg_cfg.get(method, {}) or {})
        local_field = remove_background(
            phase_combined, qsm.mask,
            voxel_size=qsm.voxel_size,
            method=method,
            **bg_kwargs,
        )
        local_field *= qsm.mask
        save_nifti(local_field, qsm.affine, qsm.header,
                   output_dir / "local_field.nii.gz")
        _diagnose("local_field", local_field, qsm.mask)

    smooth_sigma = float(cfg["dipole_inversion"].get("smooth_sigma", 0.0))
    if smooth_sigma > 0:
        from scipy import ndimage
        print(f"[main] Сглаживание σ={smooth_sigma}")
        local_field = ndimage.gaussian_filter(local_field, sigma=smooth_sigma)
        local_field *= qsm.mask

    # =====================================================================
    # ШАГ 6. Дипольная инверсия + ppm
    # =====================================================================
    print("\n=== ШАГ 6: Дипольная инверсия ===")
    inv_cfg = cfg["dipole_inversion"]
    inv_method = inv_cfg["method"]
    inv_kwargs = dict(inv_cfg.get(inv_method, {}) or {})
    if inv_method == "tikhonov" and "lambda" in inv_kwargs:
        inv_kwargs["lambda_"] = inv_kwargs.pop("lambda")

    chi = compute_qsm(
        local_field, qsm.mask,
        voxel_size=qsm.voxel_size,
        method=inv_method,
        **inv_kwargs,
    )
    _diagnose("chi (raw)", chi, qsm.mask)

    if qsm.te and any(t > 0 for t in qsm.te):
        te_arr = np.asarray(qsm.te, dtype=np.float64)
        te_eff = float((te_arr ** 2).sum() / te_arr.sum())
    else:
        te_eff = 0.02
    print(f"[QSM] TE_eff = {te_eff * 1000:.2f} ms")

    B0 = float(acq.get("B0", 3.0))
    scale = 1e6 / (GAMMA_PROTON * B0 * te_eff)
    chi_ppm = chi * scale
    chi_ppm *= qsm.mask

    mean_before = float(chi_ppm[qsm.mask].mean())
    print(f"[QSM] Смещение до: {mean_before:+.4f} ppm")
    chi_ppm = chi_ppm - mean_before
    chi_ppm *= qsm.mask
    print(f"[QSM] Смещение после: {chi_ppm[qsm.mask].mean():+.4f} ppm")

    _diagnose("chi (ppm)", chi_ppm, qsm.mask)

    save_nifti(chi_ppm, qsm.affine, qsm.header, output_dir / "qsm.nii.gz")
    save_nifti(qsm.mask.astype(np.float32), qsm.affine, qsm.header,
               output_dir / "mask.nii.gz")

    print(f"\n✅ QSM-пайплайн завершён. Результаты: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="QSM pipeline (Laplacian + RESHARP/Tikhonov). "
                    "Все пути указываются явно через CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠  АВТОМАТИЧЕСКОЕ ПЕРЕИМЕНОВАНИЕ ФАЙЛОВ
─────────────────────────────────────
Пайплайн САМ переименовывает файлы — вручную переименовывать НЕ НУЖНО.

При запуске на Шаге 1 скрипт сканирует --data-dir и переименовывает
все NIfTI-файлы в единый формат SEPIA:

  Было (любые варианты):               Стало:
  ─────────────────────────           ───────────────────────────────
  365B6A53..._701_e1.nii              echo-1_part-mag.nii
  365B6A53..._701_e1_ph.nii           echo-1_part-phase.nii
  365B6A53..._701_e1.json             echo-1_part-mag.json
  365B6A53..._701_e2.nii              echo-2_part-mag.nii
  ...                                  ...
  sub-001_echo-7_part-phase.nii       echo-7_part-phase.nii

  Поддерживаются любые имена:
    • e1.nii / e1_ph.nii
    • echo1.nii / echo1_phase.nii
    • 1.nii / 1_ph.nii
    • sub-01_echo-1_part-mag.nii
    • <любой_префикс>_eN.nii / <префикс>_eN_ph.nii

  Что произойдёт:
    1. Файлы будут ПЕРЕИМЕНОВАНЫ ВНУТРИ --data-dir (не копируются!)
    2. Создаётся карта _rename_mapping.json для отката
    3. Повторный запуск НЕ ломает уже переименованные файлы

  Откатить переименование:
    python -m preprocessing.rename_files <data-dir> --rollback

  Пробный запуск (без изменений):
    python main.py --data-dir ... --output-dir ... --dry-run

  Ожидаемые входные файлы в --data-dir ПОСЛЕ авто-переименования:
    echo-1_part-mag.nii     (и .nii.gz)
    echo-1_part-phase.nii
    echo-1_part-mag.json    ← TE (EchoTime)
    echo-2_part-mag.nii
    ...
    echo-N_part-phase.nii
    echo-N_part-mag.json

  ⚠ ВАЖНО: файлы переименовываются НА МЕСТЕ. Если хотите сохранить
           исходные имена — сделайте резервную копию папки data/nifti/
           перед первым запуском.

РЕКОМЕНДУЕМАЯ СТРУКТУРА ПАПОК
─────────────────────────────
  /путь/к/пациенту/
  ├── data/
  │   ├── nifti/           ← сюда: --data-dir
  │   │   ├── (исходные файлы .nii в любом формате)
  │   │   ├── echo-1_part-mag.nii     (после переименования)
  │   │   ├── echo-1_part-phase.nii
  │   │   └── ...
  │   └── dicom/           ← сюда: --dicom-dir (для NM-пайплайна)
  └── output/              ← сюда: --output-dir

ЧТО УКАЗЫВАТЬ В --data-dir
──────────────────────────
  ✅ ПРАВИЛЬНО: /путь/к/пациенту/data/nifti
  ❌ НЕПРАВИЛЬНО: /путь/к/пациенту/data
  ❌ НЕПРАВИЛЬНО: /путь/к/пациенту/data/nifti/echo-1_part-mag.nii

ЧТО УКАЗЫВАТЬ В --output-dir
────────────────────────────
  ✅ ПРАВИЛЬНО: /путь/к/пациенту/output
  ✅ Папка создаётся автоматически

ПРИМЕРЫ ЗАПУСКА
───────────────
  # 1. Пробный запуск (посмотреть, что будет переименовано)
  python main.py \\
      --data-dir /путь/к/пациенту/data/nifti \\
      --output-dir /путь/к/пациенту/output \\
      --dry-run

  # 2. Реальная обработка
  python main.py \\
      --data-dir /путь/к/пациенту/data/nifti \\
      --output-dir /путь/к/пациенту/output

  # 3. С готовой маской мозга (без FSL BET)
  python main.py \\
      --data-dir /путь/к/пациенту/data/nifti \\
      --output-dir /путь/к/пациенту/output \\
      --mask-path /путь/к/brain_mask.nii.gz
""",
    )
    p.add_argument("--data-dir", type=Path, required=True,
                   help="[REQUIRED] Папка с NIfTI-файлами (обычно data/nifti/). "
                        "Файлы переименуются АВТОМАТИЧЕСКИ в echo-N_part-mag/phase "
                        "(вручную не переименовывайте). "
                        "⚠ Не указывайте файл .nii — только папку.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="[REQUIRED] Папка для результатов QSM. "
                        "Создаётся автоматически.")
    p.add_argument("--config", type=Path, default=Path("config.yaml"),
                   help="Путь к config.yaml (по умолчанию: ./config.yaml)")
    p.add_argument("--mask-path", type=Path, default=None,
                   help="Готовая маска мозга (.nii.gz). "
                        "Если не указана — FSL BET построит автоматически.")
    p.add_argument("--dry-run", action="store_true",
                   help="ТОЛЬКО переименование (показать план), без обработки. "
                        "Файлы не переименовываются, только отчёт.")
    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = args.config.expanduser().resolve()
    if not cfg_path.exists():
        print(f"[FATAL] config.yaml не найден: {cfg_path}")
        print(f"        Запустите из папки с config.yaml или укажите --config")
        sys.exit(1)

    cfg = load_config(cfg_path)

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    # --- Валидация входных путей ---
    if not data_dir.exists():
        print(f"[FATAL] Папка data-dir не найдена: {data_dir}")
        print(f"        Проверьте путь. Ожидается папка с echo-*_part-mag.nii")
        sys.exit(1)

    if not data_dir.is_dir():
        print(f"[FATAL] --data-dir должен быть ПАПКОЙ, а не файлом:")
        print(f"        {data_dir}")
        print(f"        Например: --data-dir /путь/к/пациенту/data/nifti")
        sys.exit(1)

    if args.dry_run:
        cfg["preprocessing"]["dry_run"] = True

    mask_override = args.mask_path.expanduser() if args.mask_path else None

    run_pipeline(cfg, data_dir, output_dir, mask_path_override=mask_override)


if __name__ == "__main__":
    main()