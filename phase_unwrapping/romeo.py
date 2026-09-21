"""
Обёртка ROMEO через julia romeo.jl.
Находит выходной файл по факту: что появилось в папке после запуска.
"""
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import nibabel as nib


ROMEO_JL = Path(__file__).resolve().parents[1] / "romeo.jl"


def unwrap_with_romeo(phase_path: Path,
                       output_dir: Path,
                       magnitude_path: Path | None = None,
                       te: list[float] | None = None) -> Path:
    phase_path = Path(phase_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not phase_path.exists():
        raise FileNotFoundError(f"Фаза не найдена: {phase_path}")

    julia = shutil.which("julia")
    if julia is None:
        raise RuntimeError("Julia не найдена в PATH")

    if not ROMEO_JL.exists():
        raise FileNotFoundError(f"romeo.jl не найден: {ROMEO_JL}")

    # --- Снимок папки ДО ---
    files_before = {p.resolve() for p in output_dir.rglob("*") if p.is_file()}
    sizes_before = {p: p.stat().st_size for p in files_before}

    # --- Команда ---
    # ВАЖНО: -o с явным .nii.gz в конце — ROMEO обычно добавляет расширение сам,
    # но некоторые версии используют -o как ПРЕФИКС.
    out_prefix = output_dir / "unwrapped"

    cmd = [julia, str(ROMEO_JL), str(phase_path)]
    if magnitude_path:
        cmd.extend(["-m", str(Path(magnitude_path).resolve())])
    if te:
        te_str = "[" + ",".join(f"{float(t):.6f}" for t in te) + "]"
        cmd.extend(["-t", te_str])
    cmd.extend(["-o", str(out_prefix)])

    print(f"[ROMEO] Запуск: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False, capture_output=True,
                            text=True, cwd=str(output_dir))

    print(f"[ROMEO] returncode = {result.returncode}")
    if result.stdout:
        print(f"[ROMEO] STDOUT:\n{result.stdout.strip()}")
    if result.stderr:
        print(f"[ROMEO] STDERR:\n{result.stderr.strip()}")

    if result.returncode != 0:
        raise RuntimeError(f"ROMEO упал с кодом {result.returncode}")

    # --- Снимок ПОСЛЕ ---
    files_after = {p.resolve() for p in output_dir.rglob("*") if p.is_file()}
    new_files = files_after - files_before

    print(f"[ROMEO] Новых файлов: {len(new_files)}")
    for f in sorted(new_files):
        try:
            print(f"   {f.name}  ({f.stat().st_size} байт)")
        except Exception:
            print(f"   {f.name}  (?)")

    # --- Фильтр: NIfTI, не оригинальные файлы, размер > 1 МБ ---
    phase_resolved = phase_path.resolve()
    mag_resolved = Path(magnitude_path).resolve() if magnitude_path else None

    candidates = []
    for f in new_files:
        name_low = f.name.lower()
        # Должен быть NIfTI
        if not (name_low.endswith(".nii") or name_low.endswith(".nii.gz")):
            continue
        # Не оригинальные входы
        if f == phase_resolved:
            continue
        if mag_resolved and f == mag_resolved:
            continue
        # Размер > 10 МБ — реальные данные, не заголовки
        try:
            size = f.stat().st_size
        except Exception:
            continue
        if size < 10_000_000:
            print(f"[ROMEO] Пропускаю {f.name} — слишком мал ({size} байт)")
            continue
        # Приоритет файлам с "unwrapped" в имени
        priority = 0 if "unwrapped" in name_low else 1
        candidates.append((priority, -size, f))

    if candidates:
        candidates.sort()
        chosen = candidates[0][2]
        print(f"[ROMEO] ✓ Выбран: {chosen} ({chosen.stat().st_size} байт)")

        # Проверка, что это действительно NIfTI с правильной формой
        try:
            img = nib.load(str(chosen))
            print(f"[ROMEO] Форма: {img.shape}, dtype: {img.get_data_dtype()}")
        except Exception as e:
            print(f"[ROMEO] Не удалось открыть {chosen}: {e}")
            raise

        return chosen

    # Отладка
    print(f"[ROMEO] Содержимое {output_dir}:")
    for p in sorted(output_dir.iterdir()):
        try:
            print(f"   {p.name}   ({p.stat().st_size} байт)")
        except Exception:
            print(f"   {p.name}   (?)")

    raise FileNotFoundError(
        "ROMEO вернул код 0, но NIfTI-выход не найден (размером > 10 МБ). "
        "Проверьте STDOUT/STDERR выше."
    )