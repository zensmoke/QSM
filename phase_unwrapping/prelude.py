"""
Обёртка для запуска FSL PRELUDE через subprocess.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _find_fsldir() -> str:
    env_fsldir = os.environ.get("FSLDIR")
    if env_fsldir and Path(env_fsldir).is_dir():
        return env_fsldir

    candidates = [
        "/usr/local/fsl",
        "/opt/fsl",
        "/opt/homebrew/opt/fsl",
        "/Applications/fsl",
        str(Path.home() / "fsl"),
    ]
    for c in candidates:
        if (Path(c) / "bin" / "prelude").exists():
            return c
    return ""


def run_prelude(phase_path: Path,
                magnitude_path: Path,
                out_base: Path,
                mask_path: Path | None = None,
                force3d: bool = True) -> Path:
    """
    Запускает FSL PRELUDE.

    Args:
        phase_path: файл с phase (wrapped).
        magnitude_path: файл с abs (магнитуда). Обязателен.
        out_base: префикс выходного файла (без расширения).
        mask_path: опциональная маска.
        force3d: использовать -f (force3D) или -s (по срезам).

    Returns:
        Путь к развёрнутому NIfTI.
    """
    phase_path = Path(phase_path).resolve()
    magnitude_path = Path(magnitude_path).resolve()
    out_base = Path(out_base).resolve()

    fsldir = _find_fsldir()
    if not fsldir:
        raise RuntimeError("FSLDIR не найден")

    env = os.environ.copy()
    env["FSLDIR"] = fsldir
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env["PATH"] = f"{fsldir}/bin:" + env.get("PATH", "")

    cmd = [
        "prelude",
        "-p", str(phase_path),
        "-a", str(magnitude_path),   # ← КЛЮЧЕВОЕ: -a, а не -m
        "-o", str(out_base),
        "-v",
    ]
    if force3d:
        cmd.append("-f")
    else:
        cmd.append("-s")

    if mask_path is not None:
        cmd.extend(["-m", str(Path(mask_path).resolve())])

    print(f"[PRELUDE] Запуск: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, check=False, capture_output=True, text=True, env=env
    )

    if result.stdout.strip():
        print(f"[PRELUDE] STDOUT:\n{result.stdout.strip()[-1500:]}")
    if result.stderr.strip():
        print(f"[PRELUDE] STDERR:\n{result.stderr.strip()[-1500:]}")

    if result.returncode != 0:
        raise RuntimeError(
            f"PRELUDE упал (code {result.returncode}). "
            f"Смотрите STDOUT/STDERR выше."
        )

    # PRELUDE пишет <out_base>.nii.gz или <out_base>_unwrapped.nii.gz
    candidates = [
        Path(str(out_base) + ".nii.gz"),
        Path(str(out_base) + ".nii"),
        out_base.parent / "unwrapped.nii.gz",
        out_base.parent / "unwrapped_unwrapped.nii.gz",
    ]
    candidates.extend(out_base.parent.glob("*unwrapped*.nii.gz"))

    for c in candidates:
        if c.exists() and c.stat().st_size > 1_000_000:
            print(f"[PRELUDE] Результат: {c}")
            return c

    # Отладка
    print(f"[PRELUDE] Содержимое {out_base.parent}:")
    for p in sorted(out_base.parent.iterdir()):
        print(f"   {p.name}  ({p.stat().st_size} байт)")
    raise FileNotFoundError("PRELUDE не создал выходной файл")