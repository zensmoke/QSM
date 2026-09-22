"""
Извлечение демографии пациента из DICOM/JSON + нормативные значения
для разных типов последовательностей NM-MRI и напряжённостей поля.

Поддерживаемые последовательности:
  - tse        — Turbo Spin Echo (Siemens, Al Haddad 2023)
  - tse_kcl    — TSE, KCL-конвенция (Crus-mode)
  - mtc_gre    — GRE + MT-препульс (Philips)
  - gre        — обычный GRE без MT. ⚠ Не подходит для NM-CNR.
"""
from pathlib import Path
from dataclasses import dataclass
import json

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    InvalidDicomError = Exception


# =============================================================================
# СТРУКТУРА ПАЦИЕНТА
# =============================================================================

@dataclass
class PatientInfo:
    age: float | None = None
    sex: str | None = None
    patient_id: str | None = None
    study_date: str | None = None
    manufacturer: str | None = None
    field_strength: float | None = None

    def summary(self) -> str:
        parts = []
        if self.age is not None: parts.append(f"age={self.age:.0f}")
        if self.sex: parts.append(f"sex={self.sex}")
        if self.patient_id: parts.append(f"id={self.patient_id}")
        if self.field_strength: parts.append(f"B0={self.field_strength:.1f}T")
        if self.manufacturer: parts.append(f"{self.manufacturer}")
        return ", ".join(parts) if parts else "unknown"


# =============================================================================
# SD FLOOR — защита от взрыва z-score
# =============================================================================
# Биологический SD NM-CNR ~ 5–15% от mean (Al Haddad 2023, n=152).
# Если в JSON n < 20 или sd отсутствует — подставляем этот floor.
NM_SD_FLOOR_FRACTION = 0.030    # для единиц "fraction" (доля)
NM_SD_FLOOR_PERCENT  = 3.0      # для единиц "percent" (%)
MIN_N_FOR_TRUSTED_SD = 20


# =============================================================================
# НОРМАТИВНЫЕ ЗНАЧЕНИЯ (доли или проценты — см. поле "units")
# =============================================================================

NORMATIVE_CNR_BY_SEQUENCE = {
    # --- 3T TSE, KCL-конвенция (Crus-mode) — PRIMARY ---
    "3.0_tse_kcl": {
        "description": "KCL Neuromelanin-MRI, Crus-mode, TSE 3T",
        "left_sn":  {"mean": 0.2358, "sd": 0.0300},
        "right_sn": {"mean": 0.2358, "sd": 0.0300},
        "age_adjust": False,
        "units": "fraction",
    },
    # --- 3T TSE, Al Haddad 2023 (другой референс) ---
    "3.0_tse_alhaddad": {
        "description": "Al Haddad 2023 JMRI (Siemens TSE, n=152, age 53-86)",
        "left_sn":  {"mean": 10.02, "sd": 1.48},
        "right_sn": {"mean": 10.28, "sd": 1.51},
        "age_adjust": False,
        "units": "percent",
    },
    # --- Совместимость: старый ключ ---
    "3.0_tse": {
        "description": "DEPRECATED — используйте 3.0_tse_kcl или 3.0_tse_alhaddad",
        "left_sn":  {"mean": 10.02, "sd": 1.48},
        "right_sn": {"mean": 10.28, "sd": 1.51},
        "age_adjust": False,
        "units": "percent",
        "warning": "Устаревший ключ. Для KCL-протокола используйте "
                   "3.0_tse_kcl.",
    },
    # --- 3T MTC-GRE (Philips) ---
    "3.0_mtc_gre": {
        "description": "Chen 2014, Liu 2020 (Philips MTC-GRE)",
        "left_sn":  {"mean": 22.0, "sd": 4.0},
        "right_sn": {"mean": 22.0, "sd": 4.0},
        "age_adjust": False,
        "units": "percent",
    },
    # --- 3T GRE без MT ---
    "3.0_gre": {
        "description": "GRE без MT — НЕ даёт NM-контраста",
        "left_sn":  {"mean": 0.0, "sd": 5.0},
        "right_sn": {"mean": 0.0, "sd": 5.0},
        "age_adjust": False,
        "units": "percent",
        "warning": "GRE без MT не подходит для NM-CNR!",
    },
    # --- 1.5T ---
    "1.5_tse": {
        "description": "Grilo 2017 (1.5T TSE)",
        "left_sn":  {"mean": 8.0, "sd": 2.0},
        "right_sn": {"mean": 8.0, "sd": 2.0},
        "age_adjust": False,
        "units": "percent",
    },
    "1.5_mtc_gre": {
        "description": "Grilo 2017 (1.5T MTC-GRE)",
        "left_sn":  {"mean": 15.0, "sd": 5.0},
        "right_sn": {"mean": 15.0, "sd": 5.0},
        "age_adjust": False,
        "units": "percent",
    },
    # --- Fallback ---
    "fallback": {
        "description": "Неизвестный протокол — fallback на KCL 3T TSE",
        "left_sn":  {"mean": 0.2358, "sd": 0.0300},
        "right_sn": {"mean": 0.2358, "sd": 0.0300},
        "age_adjust": False,
        "units": "fraction",
    },
}


# =============================================================================
# JSON-норматив (KCL batch)
# =============================================================================

_KCL_NORMATIVE_CACHE: dict | None = None


def _json_normative_path() -> Path:
    return Path(__file__).resolve().parent / "normative_db.json"


def load_kcl_normative(force_reload: bool = False) -> dict | None:
    global _KCL_NORMATIVE_CACHE
    if _KCL_NORMATIVE_CACHE is not None and not force_reload:
        return _KCL_NORMATIVE_CACHE

    path = _json_normative_path()
    if not path.exists():
        _KCL_NORMATIVE_CACHE = None
        return None
    try:
        _KCL_NORMATIVE_CACHE = json.loads(path.read_text())
    except Exception:
        _KCL_NORMATIVE_CACHE = None
    return _KCL_NORMATIVE_CACHE


def _normative_key(field_strength: float | None, sequence: str) -> str:
    if field_strength is None:
        f = "3.0"
    elif abs(field_strength - 1.5) < 0.3:
        f = "1.5"
    else:
        f = "3.0"

    seq = (sequence or "auto").lower()
    if seq in ("tse", "auto"):
        return "tse_3t_kcl" if f == "3.0" else "tse_1.5t"
    if seq == "mtc_gre":
        return "mtc_gre_3t" if f == "3.0" else "mtc_gre_1.5t"
    if seq == "gre":
        return "gre_3t"
    return "tse_3t_kcl"


def get_normative(field_strength: float | None,
                  sequence: str = "auto") -> dict:
    """
    Возвращает норматив: сначала из JSON, потом из хардкода.

    ВАЖНО: при n < 20 или отсутствии sd подставляем floor,
    чтобы z-score не взорвался.
    """
    # --- 1. JSON ---
    json_data = load_kcl_normative()
    if json_data and "per_sequence" in json_data:
        key = _normative_key(field_strength, sequence)
        entry = json_data["per_sequence"].get(key)

        if entry and "mean" in entry:
            mean = float(entry["mean"])
            n_used = int(entry.get("n", 0))
            sd_raw = entry.get("sd")

            if sd_raw is None or n_used < MIN_N_FOR_TRUSTED_SD:
                sd = max(mean * NM_SD_FLOOR_FRACTION, 1e-4)
                note = (f"SD floor (raw={sd_raw}, n={n_used})"
                        if n_used < MIN_N_FOR_TRUSTED_SD
                        else "SD floor (raw=None)")
            else:
                sd = max(float(sd_raw), mean * 0.02)
                note = None

            return {
                "description": entry.get("source",
                                         f"KCL normative ({key})"),
                "left_sn":  {"mean": mean, "sd": sd},
                "right_sn": {"mean": mean, "sd": sd},
                "age_adjust": False,
                "units": "fraction",
                "n": n_used,
                "note": note,
            }

    # --- 2. Хардкод ---
    if field_strength is None:
        field_key = "3.0"
    elif abs(field_strength - 1.5) < 0.3:
        field_key = "1.5"
    else:
        field_key = "3.0"

    seq = (sequence or "auto").lower()
    if seq == "auto":
        seq = "tse"

    if field_key == "3.0" and seq == "tse":
        return NORMATIVE_CNR_BY_SEQUENCE["3.0_tse_kcl"]

    key = f"{field_key}_{seq}"
    if key in NORMATIVE_CNR_BY_SEQUENCE:
        return NORMATIVE_CNR_BY_SEQUENCE[key]
    return NORMATIVE_CNR_BY_SEQUENCE["fallback"]


def normalize_to_units(value: float, from_units: str, to_units: str) -> float:
    if from_units == to_units:
        return value
    if from_units == "percent" and to_units == "fraction":
        return value / 100.0
    if from_units == "fraction" and to_units == "percent":
        return value * 100.0
    return value


# =============================================================================
# АВТООПРЕДЕЛЕНИЕ ТИПА ПОСЛЕДОВАТЕЛЬНОСТИ
# =============================================================================

def detect_sequence_type(json_path: Path) -> str:
    if not json_path.exists():
        return "unknown"
    try:
        meta = json.loads(json_path.read_text())
    except Exception:
        return "unknown"

    seq = str(meta.get("ScanningSequence", "")).upper()
    var = str(meta.get("SequenceVariant", "")).upper()
    opts = str(meta.get("ScanOptions", "")).upper()
    desc = str(meta.get("SeriesDescription", "")).upper()
    proto = str(meta.get("ProtocolName", "")).upper()
    combined = f"{desc} {proto} {seq} {var} {opts}"

    if "TSE" in combined or "TURBO SPIN" in combined:
        return "tse"
    if "SE" in seq and "GR" not in seq:
        return "tse"

    has_mt = "MT" in opts or "MTC" in combined or "MT_" in combined
    has_gr = "GR" in seq or "GR" in var or "GR" in combined or "FFE" in combined
    if has_mt and has_gr:
        return "mtc_gre"
    if has_gr:
        return "gre"
    return "unknown"


# =============================================================================
# ИЗВЛЕЧЕНИЕ ИЗ DICOM / JSON
# =============================================================================

def extract_from_dicom(dicom_dir: Path) -> PatientInfo:
    info = PatientInfo()
    if not HAS_PYDICOM or not dicom_dir.exists():
        return info

    candidates = []
    for ext in ("*.dcm", "*.IMA", "*"):
        candidates.extend(dicom_dir.glob(ext))
    candidates = [f for f in candidates if f.is_file()]

    for f in candidates[:50]:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue

        info.patient_id = getattr(ds, "PatientID", None)
        info.sex = getattr(ds, "PatientSex", None)
        info.study_date = getattr(ds, "StudyDate", None)
        info.manufacturer = getattr(ds, "Manufacturer", None)
        fs = getattr(ds, "MagneticFieldStrength", None)
        if fs:
            try:
                info.field_strength = float(fs)
            except (ValueError, TypeError):
                pass

        age_str = getattr(ds, "PatientAge", None)
        if age_str:
            try:
                info.age = float(str(age_str).rstrip("YMD").strip())
            except (ValueError, AttributeError):
                pass

        if any([info.age, info.sex, info.patient_id]):
            return info
    return info


def extract_from_json(json_path: Path) -> PatientInfo:
    info = PatientInfo()
    if not json_path.exists():
        return info
    try:
        meta = json.loads(json_path.read_text())
    except Exception:
        return info

    info.sex = meta.get("PatientSex") or meta.get("Sex")
    info.patient_id = meta.get("PatientID")
    info.study_date = meta.get("AcquisitionDate") or meta.get("StudyDate")
    info.manufacturer = meta.get("Manufacturer")
    fs = meta.get("MagneticFieldStrength")
    if fs:
        try:
            info.field_strength = float(fs)
        except (ValueError, TypeError):
            pass
    age_str = meta.get("PatientAge")
    if age_str:
        try:
            info.age = float(str(age_str).rstrip("YMD").strip())
        except (ValueError, AttributeError):
            pass
    return info


def get_patient_info(data_dir: Path, json_candidates: list = None) -> PatientInfo:
    info = extract_from_dicom(data_dir)
    if info.age and info.sex and info.field_strength:
        return info

    if json_candidates is None:
        json_candidates = list(data_dir.glob("*.json"))

    for jp in json_candidates:
        j_info = extract_from_json(jp)
        if info.age is None and j_info.age is not None:
            info.age = j_info.age
        if info.sex is None and j_info.sex is not None:
            info.sex = j_info.sex
        if info.patient_id is None and j_info.patient_id is not None:
            info.patient_id = j_info.patient_id
        if info.field_strength is None and j_info.field_strength is not None:
            info.field_strength = j_info.field_strength
        if info.manufacturer is None and j_info.manufacturer is not None:
            info.manufacturer = j_info.manufacturer
    return info


# =============================================================================
# СРАВНЕНИЕ С НОРМОЙ
# =============================================================================

@dataclass
class ComparisonResult:
    measured_cnr_pct: float
    expected_cnr_pct: float
    z_score: float
    percent_deviation: float
    interpretation: str
    age_used: float | None
    sex_used: str | None
    field_used: float | None
    sequence_used: str
    normative_source: str
    warning: str | None = None


def compare_with_normative(measured_cnr_pct: float,
                            patient: PatientInfo,
                            sequence: str = "auto") -> ComparisonResult:
    """
    measured_cnr_pct — измеренный CNR в ПРОЦЕНТАХ (23.58, не 0.2358).
    """
    norm = get_normative(patient.field_strength, sequence)
    norm_units = norm.get("units", "percent")

    if norm_units == "fraction":
        measured = measured_cnr_pct / 100.0
    else:
        measured = measured_cnr_pct

    expected = norm["left_sn"]["mean"]
    sd = norm["left_sn"]["sd"] or 1e-6

    z = (measured - expected) / sd
    pct_dev = 100.0 * (measured - expected) / expected if expected else 0.0

    if z < -2.0:   interp = "abnormal_low"
    elif z < -1.5: interp = "borderline_low"
    elif z > 2.0:  interp = "abnormal_high"
    elif z > 1.5:  interp = "borderline_high"
    else:          interp = "normal"

    return ComparisonResult(
        measured_cnr_pct=measured_cnr_pct,
        expected_cnr_pct=expected * (100 if norm_units == "fraction" else 1),
        z_score=z,
        percent_deviation=pct_dev,
        interpretation=interp,
        age_used=patient.age,
        sex_used=patient.sex,
        field_used=patient.field_strength,
        sequence_used=sequence,
        normative_source=norm["description"],
        warning=norm.get("warning") or norm.get("note"),
    )