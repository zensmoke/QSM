from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Шаблоны для распознавания
# ---------------------------------------------------------------------------

# Расширения NIfTI
NII_EXT_RE = re.compile(r'\.nii(\.gz)?$', re.IGNORECASE)

# Суффикс фазы (встречается в самых разных конвенциях)
PHASE_TOKENS = ('_ph', '-ph', '_phase', '-phase',
                '_part-phase', '-part-phase', '_partphase', 'phase')

# Суффикс магнитуды (иногда встречается явно)
MAG_TOKENS = ('_mag', '-mag', '_part-mag', '-part-mag', '_partmag', 'magnitude')

# Индекс эхо: ищем одно из (в порядке приоритета):
#   echo-1, echo_1, echo1, _e1, _e01, e1, _1_, -1_ и т.д.
ECHO_PATTERNS = [
    re.compile(r'echo[-_]?(\d{1,3})', re.IGNORECASE),
    re.compile(r'[_\-]e(\d{1,3})(?=[_\-.]|$)', re.IGNORECASE),
    re.compile(r'^e(\d{1,3})(?=[_\-.]|$)', re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class EchoFile:
    echo_num: int
    magnitude: Path | None = None
    phase: Path | None = None
    json_mag: Path | None = None
    json_phase: Path | None = None


@dataclass
class DetectedSeries:
    """Одна логическая серия (набор эхо)."""
    name: str                                       # префикс серии
    echoes: dict[int, EchoFile] = field(default_factory=dict)

    def sorted_echoes(self) -> list[EchoFile]:
        return [self.echoes[k] for k in sorted(self.echoes)]

    def num_complete_echoes(self) -> int:
        """Сколько эхо имеют и магнитуду, и фазу."""
        return sum(1 for e in self.echoes.values()
                   if e.magnitude and e.phase)


# ---------------------------------------------------------------------------
# Основные функции
# ---------------------------------------------------------------------------

def detect_series(directory: str | Path) -> list[DetectedSeries]:
    """
    Сканирует папку и возвращает список обнаруженных серий QSM.

    Каждая серия — набор эхо с парными (magnitude, phase) файлами.
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Не папка: {directory}")

    nii_files = [p for p in sorted(directory.iterdir())
                 if p.is_file() and NII_EXT_RE.search(p.name)]
    json_files = {p.stem.replace('.nii', ''): p
                  for p in sorted(directory.iterdir())
                  if p.is_file() and p.suffix.lower() == '.json'}

    print(f"[detect] NIfTI-файлов: {len(nii_files)}")
    print(f"[detect] JSON-файлов:  {len(json_files)}")

    # ---- Классификация: magnitude / phase + номер эхо + префикс серии ----
    # series_key → echo_num → {'mag': Path, 'phase': Path}
    buckets: dict[str, dict[int, dict[str, Path]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for f in nii_files:
        info = _classify_file(f)
        if info is None:
            print(f"[detect] Не распознан: {f.name}")
            continue
        series_key, echo_num, kind = info
        buckets[series_key][echo_num][kind] = f

    # ---- Собираем серии ----
    series_list: list[DetectedSeries] = []
    for series_key, echo_map in buckets.items():
        s = DetectedSeries(name=series_key)
        for echo_num, kinds in echo_map.items():
            ef = EchoFile(echo_num=echo_num,
                          magnitude=kinds.get("mag"),
                          phase=kinds.get("phase"))
            # Привязка JSON
            if ef.magnitude:
                key = _nii_stem_for_json(ef.magnitude.name)
                ef.json_mag = json_files.get(key)
            if ef.phase:
                key = _nii_stem_for_json(ef.phase.name)
                ef.json_phase = json_files.get(key)
            s.echoes[echo_num] = ef
        series_list.append(s)

    # Сортировка: сначала серии с большим числом полных эхо
    series_list.sort(key=lambda s: -s.num_complete_echoes())

    print(f"[detect] Найдено серий: {len(series_list)}")
    for s in series_list:
        print(f"         серия {s.name!r}: эхо={sorted(s.echoes)}, "
              f"полных пар={s.num_complete_echoes()}")

    return series_list


def pick_best_series(series_list: list[DetectedSeries],
                     series_filter: str | None = None) -> DetectedSeries:
    """
    Выбирает наиболее подходящую серию.

    Приоритеты:
      1. Если задан series_filter — серия, чьё имя содержит подстроку.
      2. Серия с максимальным числом ПОЛНЫХ пар (mag + phase).
    """
    if not series_list:
        raise RuntimeError("Не найдено ни одной серии QSM в папке")

    if series_filter:
        for s in series_list:
            if series_filter in s.name:
                print(f"[detect] Выбрана серия по фильтру {series_filter!r}: {s.name!r}")
                return s
        print(f"[detect] Фильтр {series_filter!r} не дал результата, "
              f"беру лучшую серию")

    best = max(series_list, key=lambda s: s.num_complete_echoes())
    print(f"[detect] Выбрана серия {best.name!r} "
          f"({best.num_complete_echoes()} полных эхо)")
    return best


# ---------------------------------------------------------------------------
# Внутренние утилиты
# ---------------------------------------------------------------------------

def _classify_file(path: Path) -> tuple[str, int, str] | None:
    """
    Возвращает (series_key, echo_num, 'mag'|'phase') или None,
    если файл не распознан как QSM-эхо.
    """
    name = path.name
    stem = NII_EXT_RE.sub('', name)  # без .nii/.nii.gz

    # --- Определяем тип (mag/phase) ---
    lower = stem.lower()
    is_phase = any(tok in lower for tok in PHASE_TOKENS)
    is_mag = (not is_phase) and any(tok in lower for tok in MAG_TOKENS)
    if not is_phase and not is_mag:
        # По умолчанию: без фазового маркера считаем магнитудой
        is_phase = False
    kind = "phase" if is_phase else "mag"

    echo_num = _extract_echo_number(stem)
    if echo_num is None:
        return None

    idx = _find_echo_start(stem)
    prefix = stem[:idx].rstrip('_-.')
    series_key = prefix.lower() if prefix else "__no_prefix__"

    return series_key, echo_num, kind


def _extract_echo_number(stem: str) -> int | None:
    """Возвращает номер эхо из имени стема или None."""
    for pat in ECHO_PATTERNS:
        m = pat.search(stem)
        if m:
            return int(m.group(1))

    # Fallback: одиночное число в начале имени: '1.nii', '2_ph.nii'
    m = re.match(r'^(\d{1,3})(?:[_\-.]|$)', stem)
    if m:
        return int(m.group(1))

    return None


def _find_echo_start(stem: str) -> int:
    """Позиция в стеме, где начинается маркер эхо."""
    for pat in ECHO_PATTERNS:
        m = pat.search(stem)
        if m:
            return m.start()
    m = re.match(r'^(\d{1,3})(?:[_\-.]|$)', stem)
    if m:
        return m.start()
    return len(stem)


def _nii_stem_for_json(nii_name: str) -> str:
    """'foo_e1_ph.nii.gz' → 'foo_e1_ph' (ключ для JSON sidecar)."""
    return NII_EXT_RE.sub('', nii_name)