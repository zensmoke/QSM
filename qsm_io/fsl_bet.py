"""
Извлечение маски мозга через FSL BET.
Копирует вход в /tmp, потому что FSL не умеет работать с путями,
содержащими пробелы или не-ASCII символы.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _find_fsldir() -> str:
    """Ищет FSLDIR: сначала env, потом типичные пути."""
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
        if (Path(c) / "bin" / "bet").exists():
            return c
    return ""


def run_fsl_bet(input_path: Path,
                output_dir: Path,
                fractional_intensity: float = 0.4,
                robust: bool = True) -> Path:
    """
    Запускает FSL BET, обходя проблему пробелов в путях.

    Returns:
        Путь к сохранённой маске (*_mask.nii.gz) в output_dir.
    """
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Нет файла: {input_path}")

    # --- Окружение FSL ---
    fsldir = _find_fsldir()
    if not fsldir:
        raise RuntimeError(
            "Не найден FSLDIR. Установите переменную окружения FSLDIR "
            "или установите FSL в стандартный путь."
        )

    env = os.environ.copy()
    env["FSLDIR"] = fsldir
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env["PATH"] = f"{fsldir}/bin:" + env.get("PATH", "")

    print(f"[FSL-BET] FSLDIR = {fsldir}")

    # --- Работаем во временной папке без пробелов ---
    with tempfile.TemporaryDirectory(prefix="fslbet_") as tmpdir:
        tmp = Path(tmpdir)
        # Имя без пробелов
        safe_input = tmp / "input.nii.gz"

        # Копируем вход, при необходимости переупаковывая в .nii.gz
        shutil.copy2(input_path, safe_input)
        print(f"[FSL-BET] Скопировал вход в {safe_input}")

        out_base = tmp / "brain"

        cmd = [
            "bet",
            str(safe_input),
            str(out_base),
            "-f", str(fractional_intensity),
            "-m",
        ]
        if robust:
            cmd.extend(["-R", "-n"])

        print(f"[FSL-BET] Запуск: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, env=env,
        )

        if result.stdout.strip():
            print(f"[FSL-BET] STDOUT:\n{result.stdout.strip()}")
        if result.stderr.strip():
            print(f"[FSL-BET] STDERR:\n{result.stderr.strip()}")

        if result.returncode != 0:
            raise RuntimeError(
                f"BET упал с кодом {result.returncode}. "
                f"Смотрите STDERR выше."
            )

        # BET создаёт brain_mask.nii.gz в tmp
        tmp_mask = tmp / "brain_mask.nii.gz"
        if not tmp_mask.exists():
            cands = list(tmp.glob("*_mask.nii.gz"))
            if not cands:
                raise FileNotFoundError(
                    f"BET не создал маску. Содержимое {tmp}: "
                    f"{[p.name for p in tmp.iterdir()]}"
                )
            tmp_mask = cands[0]

        # Копируем результат в целевой output_dir
        final_mask = output_dir / "brain_mask.nii.gz"
        shutil.copy2(tmp_mask, final_mask)

        # Также скопируем brain.nii.gz (образ без черепа), если он есть
        tmp_brain = tmp / "brain.nii.gz"
        if tmp_brain.exists():
            shutil.copy2(tmp_brain, output_dir / "brain.nii.gz")

    print(f"[FSL-BET] Маска сохранена: {final_mask}")
    return final_mask