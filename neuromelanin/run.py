"""
Оркестратор NM-пайплайна: ANTs + SimpleITK + nibabel.

Запускает ТРИ процесса для избежания segfault.
Все пути — через CLI. Поддерживает TSE / MTC-GRE / GRE.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list, description: str = ""):
    if description:
        print(f"\n{'─' * 65}")
        print(f"  {description}")
        print(f"{'─' * 65}")
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ Команда упала (code {result.returncode}): {cmd[0]}")
        sys.exit(result.returncode)


def find_t1(data_dir: Path, explicit: Path | None = None) -> Path | None:
    if explicit:
        return explicit if explicit.exists() else None
    candidates = [
        data_dir / "T1_sag.nii", data_dir / "T1_sag.nii.gz",
        data_dir / "T1_tra.nii", data_dir / "T1_tra.nii.gz",
        data_dir / "T1w.nii", data_dir / "T1w.nii.gz",
    ]
    for c in candidates:
        if c.exists():
            return c
    found = sorted(data_dir.glob("T1*.nii*"))
    return found[0] if found else None


def find_nm_mri(data_dir: Path) -> Path | None:
    candidates = [
        data_dir / "echo-1_part-mag.nii",
        data_dir / "echo-1_part-mag.nii.gz",
        data_dir / "NM.nii", data_dir / "NM.nii.gz",
        data_dir / "NM_TSE.nii", data_dir / "NM_TSE.nii.gz",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    p = argparse.ArgumentParser(
        description="NM-CNR + QSM pipeline (ANTs + SimpleITK + nibabel). "
                    "Все пути указываются явно через CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠  ВАЖНО: ЗАПУСКАЙТЕ ПОСЛЕ main.py
──────────────────────────────────
Этот скрипт ОЖИДАЕТ УЖЕ ПЕРЕИМЕНОВАННЫЕ файлы в --data-dir:
    echo-1_part-mag.nii     (NM-MRI)
    echo-1_part-phase.nii
    echo-1_part-mag.json
    ...
    T1_tra.nii / T1_sag.nii

Если файлы ещё не переименованы — сначала запустите main.py:
    python main.py \\
        --data-dir /путь/к/data/nifti \\
        --output-dir /путь/к/output

main.py автоматически переименует ВСЕ NIfTI-файлы в формат
echo-N_part-mag/phase. Вручную переименовывать НЕ НУЖНО.

Если у вас уже есть готовые NM-MRI (TSE) — файл называется иначе
(например, NM.nii). Укажите его явно:
    --nm-mri /путь/к/data/nifti/NM.nii

ОЖИДАЕМЫЕ ФАЙЛЫ В --data-dir
─────────────────────────────
  Для MTC-GRE (Philips) — ваш текущий протокол:
    echo-1_part-mag.nii     ← NM-MRI = echo-1 с MTC-препульсом
    echo-1_part-phase.nii
    echo-1_part-mag.json
    ...
    T1_tra.nii или T1_sag.nii

  Для TSE (Siemens) — новый протокол:
    NM.nii или NM_TSE.nii   ← отдельный TSE-скан
    T1_tra.nii или T1_sag.nii

ТИПЫ NM-MRI ПОСЛЕДОВАТЕЛЬНОСТЕЙ
────────────────────────────────
  tse       — Turbo Spin Echo (Siemens, Al Haddad 2023)
              Норма CNR SN: ~10%. Не совместим с QSM.
  mtc_gre   — GRE + MT-препульс (Philips)
              Норма CNR SN: ~22%. Совместим с QSM.
  gre       — GRE без MT. ⚠ Не подходит для NM-CNR.
  auto      — определить из JSON sidecar (по умолчанию)

СТРУКТУРА ПАПОК
───────────────
  /путь/к/пациенту/
  ├── data/
  │   ├── nifti/           ← --data-dir
  │   │   ├── echo-1_part-mag.nii     (MTC-GRE) ИЛИ NM.nii (TSE)
  │   │   ├── echo-1_part-phase.nii
  │   │   ├── echo-1_part-mag.json
  │   │   ├── ... (остальные эхо)
  │   │   └── T1_tra.nii или T1_sag.nii
  │   └── dicom/           ← --dicom-dir
  └── output/              ← --output-dir (должна содержать qsm.nii.gz)

ПРИМЕРЫ
───────
  # MTC-GRE (Philips) — стандартный протокол
  python neuromelanin/run.py \\
      --data-dir /путь/data/nifti \\
      --dicom-dir /путь/data/dicom \\
      --output-dir /путь/output \\
      --kcl-dir /путь/KCL \\
      --syn

  # TSE (Siemens) — с явным файлом NM-MRI
  python neuromelanin/run.py \\
      --data-dir /путь/data/nifti \\
      --dicom-dir /путь/data/dicom \\
      --output-dir /путь/output \\
      --kcl-dir /путь/KCL \\
      --nm-mri /путь/data/nifti/NM_TSE.nii \\
      --nm-sequence tse \\
      --syn
""",
    )

    # Обязательные пути
    p.add_argument("--data-dir", type=Path, required=True,
                   help="[REQUIRED] Папка с УЖЕ ПЕРЕИМЕНОВАННЫМИ NIfTI "
                        "(echo-N_part-mag/phase). "
                        "Переименование делает main.py — запустите его первым. "
                        "⚠ Не указывайте файл .nii — только папку.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="[REQUIRED] Папка результатов (внутри — nm_analysis/). "
                        "Должна содержать qsm.nii.gz (результат main.py) "
                        "для QSM-части анализа.")
    p.add_argument("--kcl-dir", type=Path, required=True,
                   help="[REQUIRED] Папка с KCL атласом (содержит templates/). "
                        "⚠ Не указывайте templates/ напрямую.")

    # Опциональные пути
    p.add_argument("--dicom-dir", type=Path, default=None,
                   help="Папка с DICOM (демография; обычно data/dicom/)")
    p.add_argument("--t1", type=Path, default=None,
                   help="Явный путь к T1 (иначе — автопоиск T1_tra/T1_sag/T1w)")
    p.add_argument("--nm-mri", type=Path, default=None,
                   help="Явный путь к NM-MRI. "
                        "По умолчанию: echo-1_part-mag.nii (MTC-GRE). "
                        "Для TSE укажите: NM.nii или NM_TSE.nii.")
    p.add_argument("--qsm", type=Path, default=None,
                   help="Явный путь к QSM (по умолчанию output-dir/qsm.nii.gz)")

    # Параметры NM
    p.add_argument("--nm-sequence", type=str, default="auto",
                   choices=["auto", "tse", "mtc_gre", "gre"],
                   help="Тип NM-MRI последовательности: "
                        "tse | mtc_gre | gre | auto (default: auto). "
                        "Влияет на выбор нормативных значений CNR.")

    # Параметры обработки
    p.add_argument("--syn", action="store_true",
                   help="SyN для T1→MNI (иначе Affine). "
                        "⚠ Не указан → Affine может провалиться — используйте --syn.")
    p.add_argument("--force", action="store_true",
                   help="Пересчитать всё с нуля (удалить старые маски и регистрацию)")
    p.add_argument("--no-montage", action="store_true",
                   help="Без PNG-монтажа")
    p.add_argument("--threads", type=int, default=4,
                   help="Потоки для ANTs (default: 4)")
    p.add_argument("--field-strength", type=float, default=None,
                   help="B0 в Тесла (1.5 или 3.0). Если не указан — из DICOM.")
    p.add_argument("--erode-sn", type=int, default=3,
                   help="Радиус эрозии маски SN (default: 3).")

    # Сдвиг SN
    p.add_argument("--sn-shift", type=int, default=None,
                   help="Фиксированный сдвиг SN в вокселях "
                        "(если не задан --auto-sn-shift)")
    p.add_argument("--auto-sn-shift", action="store_true", default=True,
                   help="Автоматическая калибровка сдвига SN (default: True)")
    p.add_argument("--no-auto-sn-shift", dest="auto_sn_shift",
                   action="store_false",
                   help="Отключить автокалибровку (использовать --sn-shift)")
    p.add_argument("--sn-shift-range", type=str, default="0,30,2",
                   help="Диапазон автокалибровки: start,stop,step "
                        "(default: '0,30,2')")

    args = p.parse_args()

    DATA_DIR = args.data_dir.expanduser().resolve()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    KCL_DIR = args.kcl_dir.expanduser().resolve()
    DICOM_DIR = args.dicom_dir.expanduser().resolve() if args.dicom_dir else None

    WORK_DIR = OUTPUT_DIR / "nm_analysis"
    REG_DIR = WORK_DIR / "_reg"
    ATLAS_NATIVE = REG_DIR / "atlas_native.nii.gz"
    REG_META = REG_DIR / "reg_meta.json"

    ATLAS_MNI = KCL_DIR / "templates" / "midbrain_atlas_space-MNI152NLin2009cSym.nii.gz"
    MNI_T1 = KCL_DIR / "templates" / "mni_icbm152_t1_tal_nlin_sym_09c.nii"

    NM_MRI = args.nm_mri.expanduser() if args.nm_mri else find_nm_mri(DATA_DIR)
    T1_PATH = find_t1(DATA_DIR, args.t1)
    QSM_PATH = args.qsm.expanduser() if args.qsm else (OUTPUT_DIR / "qsm.nii.gz")

    MASK_FILES = [
        WORK_DIR / "SN_VTA_L_native.nii.gz",
        WORK_DIR / "SN_VTA_R_native.nii.gz",
        WORK_DIR / "CrusCerebri_native.nii.gz",
    ]

    script_dir = Path(__file__).parent

    if args.force:
        print("[orchestrator] Очистка кэша (--force)...")
        ATLAS_NATIVE.unlink(missing_ok=True)
        REG_META.unlink(missing_ok=True)
        for pat in ["SN_VTA*.nii.gz", "CrusCerebri*.nii.gz",
                     "all_masks_combined.nii.gz", "sn_shift_calibration.json"]:
            for f in WORK_DIR.glob(pat):
                f.unlink()

    # Проверки
    print("\n" + "=" * 65)
    print("  NM-CNR PIPELINE")
    print("=" * 65)
    print(f"\n  Data dir:     {DATA_DIR}")
    print(f"  DICOM dir:    {DICOM_DIR if DICOM_DIR and DICOM_DIR.exists() else 'не указана'}")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print(f"  KCL dir:      {KCL_DIR}")

    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        print(f"\n❌ --data-dir не папка: {DATA_DIR}")
        sys.exit(1)
    if not KCL_DIR.exists() or not KCL_DIR.is_dir():
        print(f"\n❌ --kcl-dir не папка: {KCL_DIR}")
        sys.exit(1)

    if NM_MRI is None or not NM_MRI.exists():
        print(f"\n❌ NM-MRI не найден в {DATA_DIR}")
        sys.exit(1)
    print(f"\n  NM-MRI:       {NM_MRI.name}")

    print(f"  QSM:          {QSM_PATH.name if QSM_PATH.exists() else '⚠ не найден'}")

    if T1_PATH is None or not T1_PATH.exists():
        print(f"\n❌ T1 не найден в {DATA_DIR}")
        sys.exit(1)
    print(f"  T1:           {T1_PATH.name}")
    print(f"  NM sequence:  {args.nm_sequence}")

    if args.field_strength:
        print(f"  B0:           {args.field_strength:.1f}T")

    print(f"  SyN:          {args.syn}")
    print(f"  Erode SN:     {args.erode_sn}")
    if args.auto_sn_shift:
        print(f"  SN shift:     АВТО ({args.sn_shift_range})")
    else:
        print(f"  SN shift:     фикс. {args.sn_shift or 0}")

    if not ATLAS_MNI.exists() or not MNI_T1.exists():
        print(f"\n❌ Шаблоны KCL не найдены в {KCL_DIR}/templates/")
        sys.exit(1)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    REG_DIR.mkdir(parents=True, exist_ok=True)

    # ПРОЦЕСС 1: ANTs
    if not ATLAS_NATIVE.exists():
        run(
            [sys.executable, str(script_dir / "register.py"),
             "--subject-t1", str(T1_PATH),
             "--nm-mri", str(NM_MRI),
             "--mni-t1", str(MNI_T1),
             "--atlas-mni", str(ATLAS_MNI),
             "--output", str(ATLAS_NATIVE),
             "--meta", str(REG_META),
             "--type", "SyN" if args.syn else "Affine",
             "--threads", str(args.threads)],
            description="ПРОЦЕСС 1: ANTs регистрация"
        )
    else:
        print(f"\n[orchestrator] Регистрация в кэше")

    # =========================================================================
    # ПРОЦЕСС 1.5: ИЗВЛЕЧЕНИЕ МАСОК
    # =========================================================================
    masks_exist = all(p.exists() for p in MASK_FILES)

    # Авто-инвалидация: если атлас новее масок — пересоздать маски
    if masks_exist and ATLAS_NATIVE.exists():
        atlas_mtime = ATLAS_NATIVE.stat().st_mtime
        oldest_mask = min(p.stat().st_mtime for p in MASK_FILES)
        if atlas_mtime > oldest_mask:
            print(f"\n[orchestrator] Атлас новее масок — "
                  f"пересоздаю маски")
            for p in MASK_FILES:
                p.unlink(missing_ok=True)
            masks_exist = False

    if not masks_exist or args.force:
        cmd = [
            sys.executable, str(script_dir / "extract_masks.py"),
            "--atlas-native", str(ATLAS_NATIVE),
            "--out-dir", str(WORK_DIR),
            "--erode-sn", str(args.erode_sn),
        ]
        if QSM_PATH.exists():
            cmd.extend(["--reference", str(QSM_PATH)])
        else:
            cmd.extend(["--reference", str(NM_MRI)])

        if args.auto_sn_shift:
            cmd.append("--auto-sn-shift")
            cmd.extend(["--sn-shift-range", args.sn_shift_range])
            desc = "ПРОЦЕСС 1.5: Извлечение масок (автокалибровка)"
        else:
            cmd.extend(["--sn-shift", str(args.sn_shift or 0)])
            desc = f"ПРОЦЕСС 1.5: Извлечение масок (сдвиг {args.sn_shift or 0})"

        run(cmd, description=desc)
    else:
        print(f"\n[orchestrator] Маски в кэше")

    # ПРОЦЕСС 2: анализ
    cmd = [
        sys.executable, str(script_dir / "analyze.py"),
        "--work-dir", str(WORK_DIR),
        "--qsm", str(QSM_PATH),
        "--mag-mean", str(NM_MRI),
        "--nm-mri", str(NM_MRI),
        "--atlas-native", str(ATLAS_NATIVE),
        "--reg-meta", str(REG_META),
        "--nm-sequence", args.nm_sequence,
    ]
    if DICOM_DIR and DICOM_DIR.exists():
        cmd.extend(["--dicom-dir", str(DICOM_DIR)])
    if args.field_strength:
        cmd.extend(["--field-strength", str(args.field_strength)])
    if args.no_montage:
        cmd.append("--no-montage")

    run(cmd, description="ПРОЦЕСС 2: Анализ")

    print("\n" + "=" * 65)
    print("  ГОТОВО")
    print("=" * 65)
    print(f"\n  Результаты: {WORK_DIR}")


if __name__ == "__main__":
    main()