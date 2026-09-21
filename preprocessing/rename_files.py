"""
Переименование QSM-файлов IN-PLACE в формат SEPIA.

Файлы не копируются — переименовываются прямо в исходной папке.
Для отката сохраняется карта _rename_mapping.json.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from preprocessing.auto_detect import (
    DetectedSeries, detect_series, pick_best_series,
)

MAPPING_FILE = "_rename_mapping.json"


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def rename_qsm_files(directory: str | Path,
                     series_filter: str | None = None,
                     dry_run: bool = False,
                     backup_before: bool = False) -> dict:
    """
    Переименовывает QSM-файлы в папке directory в формат SEPIA:
        echo-N_part-mag.nii / .nii.gz
        echo-N_part-phase.nii / .nii.gz
        echo-N_part-mag.json
        echo-N_part-phase.json

    Args:
        directory: папка с QSM-данными (файлы переименовываются здесь же)
        series_filter: подстрока для выбора серии (None = авто)
        dry_run: True — только показать, ничего не менять
        backup_before: True — сделать бэкап имён в JSON перед переименованием

    Returns:
        dict с ключами 'mag', 'phase', 'json_mag', 'json_phase' (пути после переименования)
        и 'mapping' — {новое_имя: старое_имя}
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Не папка: {directory}")

    print(f"\n[rename] Папка: {directory}")
    print(f"[rename] Режим: {'DRY-RUN (без изменений)' if dry_run else 'IN-PLACE'}")

    series_list = detect_series(directory)
    series = pick_best_series(series_list, series_filter=series_filter)

    # Собираем план переименования
    plan: list[tuple[Path, Path]] = []

    for ef in series.sorted_echoes():
        n = ef.echo_num

        if ef.magnitude:
            dst = directory / f"echo-{n}_part-mag{_ext(ef.magnitude)}"
            plan.append((ef.magnitude, dst))
        if ef.phase:
            dst = directory / f"echo-{n}_part-phase{_ext(ef.phase)}"
            plan.append((ef.phase, dst))
        if ef.json_mag:
            dst = directory / f"echo-{n}_part-mag.json"
            plan.append((ef.json_mag, dst))
        if ef.json_phase:
            dst = directory / f"echo-{n}_part-phase.json"
            plan.append((ef.json_phase, dst))

    # Проверка конфликтов
    _check_conflicts(plan, directory)

    if dry_run:
        print("\n[rename] План переименования:")
        for src, dst in plan:
            mark = "  (уже так)" if src == dst else ""
            print(f"   {src.name}  ->  {dst.name}{mark}")
        print(f"[rename] Всего: {len(plan)} файлов")
        return {"mag": [], "phase": [], "json_mag": [], "json_phase": [],
                "mapping": {}, "dry_run": True}

    # -----------------------------------------------------------------
    # In-place переименование
    # -----------------------------------------------------------------
    mapping: dict[str, str] = {}   # new_name -> old_name
    renamed = {"mag": [], "phase": [], "json_mag": [], "json_phase": []}

    for src, dst in plan:
        if src == dst:
            print(f"[rename] Без изменений: {src.name}")
            # Всё равно добавляем в mapping для полноты
            mapping[dst.name] = src.name
        else:
            src.rename(dst)
            print(f"[rename] {src.name}  ->  {dst.name}")
            mapping[dst.name] = src.name

        # Записываем в соответствующую категорию
        name = dst.name
        if name.endswith("_part-mag.nii") or name.endswith("_part-mag.nii.gz"):
            renamed["mag"].append(dst)
        elif name.endswith("_part-phase.nii") or name.endswith("_part-phase.nii.gz"):
            renamed["phase"].append(dst)
        elif name.endswith("_part-mag.json"):
            renamed["json_mag"].append(dst)
        elif name.endswith("_part-phase.json"):
            renamed["json_phase"].append(dst)

    # Сортировка по номеру эхо
    def key(p):
        import re
        m = re.search(r'echo-(\d+)', p.name)
        return int(m.group(1)) if m else 0
    for k in renamed:
        renamed[k].sort(key=key)

    # Сохраняем карту для отката
    _save_mapping(directory, mapping)
    renamed["mapping"] = mapping

    print(f"\n[rename] Готово: mag={len(renamed['mag'])}, "
          f"phase={len(renamed['phase'])}, "
          f"json_mag={len(renamed['json_mag'])}, "
          f"json_phase={len(renamed['json_phase'])}")
    print(f"[rename] Карта отката: {directory / MAPPING_FILE}")

    return renamed


# ---------------------------------------------------------------------------
# Откат
# ---------------------------------------------------------------------------

def rollback_rename(directory: str | Path) -> None:
    """
    Восстанавливает исходные имена файлов, используя карту _rename_mapping.json.
    Удаляет карту после успешного отката.
    """
    directory = Path(directory).expanduser().resolve()
    map_path = directory / MAPPING_FILE

    if not map_path.exists():
        print(f"[rollback] Карта не найдена: {map_path}")
        return

    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: dict[str, str] = data.get("mapping", {})
    if not mapping:
        print("[rollback] Карта пуста")
        return

    print(f"[rollback] Восстанавливаю {len(mapping)} файлов в {directory}")
    restored = 0

    for new_name, old_name in mapping.items():
        src = directory / new_name
        dst = directory / old_name
        if not src.exists():
            print(f"[rollback] Пропущен (нет файла): {new_name}")
            continue
        if dst.exists() and dst != src:
            print(f"[rollback] Конфликт: {old_name} уже существует — пропуск")
            continue
        if src == dst:
            continue
        src.rename(dst)
        print(f"[rollback] {new_name}  ->  {old_name}")
        restored += 1

    map_path.unlink()
    print(f"[rollback] Восстановлено: {restored}, карта удалена")


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

def _ext(path: Path) -> str:
    return '.nii.gz' if path.name.lower().endswith('.nii.gz') else '.nii'


def _check_conflicts(plan: list[tuple[Path, Path]],
                     directory: Path) -> None:
    """Проверяет, не помешает ли переименование существующим файлам."""
    plan_srcs = {src.resolve() for src, _ in plan}
    plan_dsts: dict[Path, Path] = {}
    for src, dst in plan:
        dst_res = dst.resolve()
        if dst_res in plan_dsts and plan_dsts[dst_res] != src.resolve():
            raise RuntimeError(
                f"Конфликт: два файла претендуют на имя {dst.name}: "
                f"{plan_dsts[dst_res].name} и {src.name}"
            )
        plan_dsts[dst_res] = src.resolve()

    for src, dst in plan:
        if dst.exists() and dst.resolve() not in plan_srcs:
            raise RuntimeError(
                f"Файл {dst.name} уже существует и не будет переименован. "
                f"Удалите его или используйте другой series_filter."
            )


def _save_mapping(directory: Path, mapping: dict[str, str]) -> None:
    map_path = directory / MAPPING_FILE
    payload = {
        "timestamp": datetime.now().isoformat(),
        "directory": str(directory),
        "mapping": mapping,
    }
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Работа с JSON sidecar
# ---------------------------------------------------------------------------

def read_te_from_json(json_path: str | Path) -> float | None:
    """
    Извлекает EchoTime (в секундах) из JSON sidecar.
    Поддерживает разные имена ключей и авто-перевод мс → сек.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"[json] Не удалось прочитать {json_path}: {e}")
        return None

    for key in ("EchoTime", "echo_time", "EchoTimeSeconds", "TE", "EchoTimes"):
        if key in meta:
            val = meta[key]
            if isinstance(val, (list, tuple)):
                val = val[0]
            try:
                te = float(val)
            except (TypeError, ValueError):
                continue
            if te > 1.0:          # мс → сек
                te /= 1000.0
            return te
    return None


def read_meta_from_json(json_path: str | Path) -> dict:
    """Читает все метаданные из JSON sidecar (или пустой dict)."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QSM rename tool (in-place)")
    parser.add_argument("directory", help="Папка с QSM-данными")
    parser.add_argument("--series-filter", default=None,
                        help="Подстрока для выбора серии (например, '701')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Только показать план, ничего не менять")
    parser.add_argument("--rollback", action="store_true",
                        help="Откатить переименование по карте")
    args = parser.parse_args()

    if args.rollback:
        rollback_rename(args.directory)
    else:
        rename_qsm_files(
            args.directory,
            series_filter=args.series_filter,
            dry_run=args.dry_run,
        )