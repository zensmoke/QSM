"""
Оркестратор NM-пайплайна: ANTs + SimpleITK + nibabel.

Запускает ТРИ процесса:
  1) ANTs регистрация (standard / intermediate / direct / multi-t1)
  2) SimpleITK: извлечение масок
  3) nibabel: анализ NM-CNR + QSM
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
        data_dir / "T1_cor.nii", data_dir / "T1_cor.nii.gz",
    ]
    for c in candidates:
        if c.exists():
            return c
    found = sorted(data_dir.glob("T1*.nii*"))
    return found[0] if found else None


def find_t1_planes(data_dir: Path) -> dict:
    planes = {}
    for p in sorted(data_dir.glob("T1*.nii*")):
        name = p.name.lower()
        if "sag" in name:
            planes.setdefault("sag", p)
        elif "tra" in name:
            planes.setdefault("tra", p)
        elif "cor" in name:
            planes.setdefault("cor", p)
        else:
            planes.setdefault(p.stem, p)
    return planes


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
        description="NM-CNR + QSM pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
РЕЖИМЫ РЕГИСТРАЦИИ NM → MNI
══════════════════════════
  1) Standard (по умолчанию):
       --t1 T1_sag.nii
     NM --Rigid--> T1_sag --SyN--> MNI

  2) Intermediate через вторую T1-плоскость:
       --t1 T1_sag.nii --t1-other T1_tra.nii
     NM --Rigid--> T1_tra --Rigid--> T1_sag --SyN--> MNI

  3) Direct (без T1):
       --direct-nm-to-mni --direct-method Affine
     NM --Affine--> MNI

  4) Мульти-T1 (перебор всех стратегий):
       --multi-t1
     Пробует все standard_*, intermediate_*, direct_* и выбирает
     лучшую по SN χ mean в QSM. Требует qsm.nii.gz.

ПРИМЕРЫ
═══════
  # Мульти-T1
  python neuromelanin/run.py \\
      --data-dir ... --output-dir ... --kcl-dir ./KCL \\
      --multi-t1 --force

  # Intermediate через T1_tra
  python neuromelanin/run.py \\
      --data-dir ... --output-dir ... --kcl-dir ./KCL \\
      --t1 T1_sag.nii --t1-other T1_tra.nii --force

  # Direct
  python neuromelanin/run.py \\
      --data-dir ... --output-dir ... --kcl-dir ./KCL \\
      --direct-nm-to-mni --direct-method Affine --force
""",
    )

    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--kcl-dir", type=Path, required=True)

    p.add_argument("--dicom-dir", type=Path, default=None)
    p.add_argument("--t1", type=Path, default=None)
    p.add_argument("--t1-other", type=Path, default=None,
                   help="Вторая T1-плоскость (не anchor) для intermediate.")
    p.add_argument("--nm-mri", type=Path, default=None)
    p.add_argument("--qsm", type=Path, default=None)

    p.add_argument("--nm-sequence", type=str, default="auto",
                   choices=["auto", "tse", "mtc_gre", "gre"])

    p.add_argument("--syn", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-montage", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--field-strength", type=float, default=None)
    p.add_argument("--erode-sn", type=int, default=3)

    p.add_argument("--sn-shift", type=int, default=None)
    p.add_argument("--auto-sn-shift", action="store_true", default=True)
    p.add_argument("--no-auto-sn-shift", dest="auto_sn_shift",
                   action="store_false")
    p.add_argument("--sn-shift-range", type=str, default="0,30,2")

    p.add_argument("--direct-nm-to-mni", action="store_true")
    p.add_argument("--direct-method", default="Affine",
                   choices=["Affine", "Rigid", "SyN"])

    p.add_argument("--multi-t1", action="store_true")
    p.add_argument("--multi-keep-all", action="store_true")
    p.add_argument("--multi-strategies", type=str, default=None)
    p.add_argument("--multi-timeout", type=int, default=600)

    args = p.parse_args()

    DATA_DIR = args.data_dir.expanduser().resolve()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    KCL_DIR = args.kcl_dir.expanduser().resolve()
    DICOM_DIR = (args.dicom_dir.expanduser().resolve()
                 if args.dicom_dir else None)

    WORK_DIR = OUTPUT_DIR / "nm_analysis"
    REG_DIR = WORK_DIR / "_reg"
    ATLAS_NATIVE = REG_DIR / "atlas_native.nii.gz"
    REG_META = REG_DIR / "reg_meta.json"

    ATLAS_MNI = KCL_DIR / "templates" / "midbrain_atlas_space-MNI152NLin2009cSym.nii.gz"
    MNI_T1 = KCL_DIR / "templates" / "mni_icbm152_t1_tal_nlin_sym_09c.nii"

    NM_MRI = args.nm_mri.expanduser() if args.nm_mri else find_nm_mri(DATA_DIR)
    T1_PATH = find_t1(DATA_DIR, args.t1)
    QSM_PATH = (args.qsm.expanduser() if args.qsm
                else (OUTPUT_DIR / "qsm.nii.gz"))

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
        import shutil as _sh
        for dname in ("debug_direct", "debug_intermediate",
                      "multi_strategies"):
            d = REG_DIR / dname
            if d.exists():
                _sh.rmtree(d, ignore_errors=True)

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
    print(f"  QSM:          "
          f"{QSM_PATH.name if QSM_PATH.exists() else '⚠ не найден'}")

    if args.multi_t1:
        planes = find_t1_planes(DATA_DIR)
        if not planes:
            print(f"\n❌ --multi-t1: T1_*.nii* не найдены в {DATA_DIR}")
            sys.exit(1)
        if not QSM_PATH.exists():
            print(f"\n❌ --multi-t1 требует QSM: {QSM_PATH}")
            sys.exit(1)
        print(f"  Регистрация:  МУЛЬТИ-T1 ({len(planes)} плоскостей: "
              f"{', '.join(sorted(planes))})")
    elif args.direct_nm_to_mni:
        print(f"  T1:           (не используется)")
        print(f"  Регистрация:  ПРЯМАЯ NM → MNI ({args.direct_method})")
    else:
        if T1_PATH is None or not T1_PATH.exists():
            print(f"\n❌ T1 не найден в {DATA_DIR}")
            sys.exit(1)
        print(f"  T1 (anchor):  {T1_PATH.name}")
        if args.t1_other:
            print(f"  T1 (other):   {args.t1_other.name}")
            print(f"  Регистрация:  INTERMEDIATE через T1_other")
        else:
            print(f"  SyN:          {args.syn}")
            print(f"  Регистрация:  STANDARD")

    print(f"  NM sequence:  {args.nm_sequence}")
    if args.field_strength:
        print(f"  B0:           {args.field_strength:.1f}T")
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

    # =====================================================================
    # ПРОЦЕСС 1: Регистрация
    # =====================================================================
    if not ATLAS_NATIVE.exists():

        if args.multi_t1:
            multi_script = script_dir / "register_multi.py"
            if not multi_script.exists():
                print(f"\n❌ Не найден {multi_script}")
                sys.exit(1)

            cmd = [
                sys.executable, str(multi_script),
                "--nm-mri", str(NM_MRI),
                "--qsm", str(QSM_PATH),
                "--mni-t1", str(MNI_T1),
                "--atlas-mni", str(ATLAS_MNI),
                "--data-dir", str(DATA_DIR),
                "--output", str(ATLAS_NATIVE),
                "--meta", str(REG_META),
                "--register-script", str(script_dir / "register.py"),
                "--threads", str(args.threads),
                "--timeout-per-strategy", str(args.multi_timeout),
            ]
            if args.multi_keep_all:
                cmd.append("--keep-all")
            if args.multi_strategies:
                cmd.extend(["--strategies", args.multi_strategies])

            run(cmd, description="ПРОЦЕСС 1: Мульти-T1 регистрация")

        elif args.direct_nm_to_mni:
            cmd = [
                sys.executable, str(script_dir / "register.py"),
                "--subject-t1", "none",
                "--nm-mri", str(NM_MRI),
                "--mni-t1", str(MNI_T1),
                "--atlas-mni", str(ATLAS_MNI),
                "--output", str(ATLAS_NATIVE),
                "--meta", str(REG_META),
                "--type", "SyN",
                "--threads", str(args.threads),
                "--direct-nm-to-mni",
                "--direct-method", args.direct_method,
            ]
            run(cmd, description=(
                f"ПРОЦЕСС 1: ANTs (ПРЯМАЯ NM→MNI, {args.direct_method})"
            ))

        elif args.t1_other:
            cmd = [
                sys.executable, str(script_dir / "register.py"),
                "--subject-t1", str(T1_PATH),
                "--t1-anchor", str(T1_PATH),
                "--t1-other", str(args.t1_other.expanduser()),
                "--nm-mri", str(NM_MRI),
                "--mni-t1", str(MNI_T1),
                "--atlas-mni", str(ATLAS_MNI),
                "--output", str(ATLAS_NATIVE),
                "--meta", str(REG_META),
                "--type", "SyN",
                "--threads", str(args.threads),
            ]
            run(cmd, description="ПРОЦЕСС 1: ANTs (intermediate)")

        else:
            cmd = [
                sys.executable, str(script_dir / "register.py"),
                "--subject-t1", str(T1_PATH),
                "--nm-mri", str(NM_MRI),
                "--mni-t1", str(MNI_T1),
                "--atlas-mni", str(ATLAS_MNI),
                "--output", str(ATLAS_NATIVE),
                "--meta", str(REG_META),
                "--type", "SyN" if args.syn else "Affine",
                "--threads", str(args.threads),
            ]
            run(cmd, description="ПРОЦЕСС 1: ANTs (standard)")

    else:
        print(f"\n[orchestrator] Регистрация в кэше")

    # =====================================================================
    # ПРОЦЕСС 1.5: Извлечение масок
    # =====================================================================
    masks_exist = all(p.exists() for p in MASK_FILES)

    if masks_exist and ATLAS_NATIVE.exists():
        atlas_mtime = ATLAS_NATIVE.stat().st_mtime
        oldest_mask = min(p.stat().st_mtime for p in MASK_FILES)
        if atlas_mtime > oldest_mask:
            print(f"\n[orchestrator] Атлас новее масок — пересоздаю маски")
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

    # =====================================================================
    # ПРОЦЕСС 2: Анализ
    # =====================================================================
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