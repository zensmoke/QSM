"""
Извлечение демографии пациента из DICOM/JSON + нормативные значения
для разных типов последовательностей NM-MRI и напряжённостей поля.

Поддерживаемые последовательности:
  - tse        — Turbo Spin Echo (Siemens, Al Haddad 2023). CNR ~10%
  - mtc_gre    — GRE + MT-препульс (Philips). CNR ~22%
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
# НОРМАТИВНЫЕ ЗНАЧЕНИЯ ПО ПОСЛЕДОВАТЕЛЬНОСТИ И ПОЛЮ
# =============================================================================

NORMATIVE_CNR_BY_SEQUENCE = {
    # --- 3T ---
    "3.0_tse": {
        "description": "Al Haddad et al. 2023, JMRI (Siemens TSE, n=152, age 53-86)",
        "left_sn":  {"mean": 10.02, "sd": 1.48},
        "right_sn": {"mean": 10.28, "sd": 1.51},
        "age_adjust": False,
    },
    "3.0_mtc_gre": {
        "description": "Chen 2014, Liu 2020 (Philips MTC-GRE, ориентировочно)",
        "left_sn":  {"mean": 22.0, "sd": 4.0},
        "right_sn": {"mean": 22.0, "sd": 4.0},
        "age_adjust": False,
    },
    "3.0_gre": {
        "description": "GRE без MT — НЕ даёт NM-контраста",
        "left_sn":  {"mean": 0.0, "sd": 5.0},
        "right_sn": {"mean": 0.0, "sd": 5.0},
        "age_adjust": False,
        "warning": "GRE без MT не подходит для NM-CNR! "
                   "Используйте TSE или MTC-GRE.",
    },
    # --- 1.5T ---
    "1.5_tse": {
        "description": "Grilo 2017 (1.5T TSE, ориентировочно)",
        "left_sn":  {"mean": 8.0, "sd": 2.0},
        "right_sn": {"mean": 8.0, "sd": 2.0},
        "age_adjust": False,
    },
    "1.5_mtc_gre": {
        "description": "Grilo 2017 (1.5T MTC-GRE, ориентировочно)",
        "left_sn":  {"mean": 15.0, "sd": 5.0},
        "right_sn": {"mean": 15.0, "sd": 5.0},
        "age_adjust": False,
    },
    # --- Fallback ---
    "fallback": {
        "description": "Неизвестная последовательность — fallback на TSE 3T",
        "left_sn":  {"mean": 10.02, "sd": 3.5},
        "right_sn": {"mean": 10.28, "sd": 3.6},
        "age_adjust": False,
    },
}


def get_normative(field_strength: float | None, sequence: str = "auto") -> dict:
    """Возвращает нормативные значения для поля и последовательности."""
    # Нормализуем поле
    if field_strength is None:
        field_key = "3.0"
    elif abs(field_strength - 1.5) < 0.3:
        field_key = "1.5"
    else:
        field_key = "3.0"

    # Нормализуем последовательность
    seq = (sequence or "auto").lower()
    if seq == "auto":
        seq = "tse"  # fallback

    key = f"{field_key}_{seq}"
    if key in NORMATIVE_CNR_BY_SEQUENCE:
        return NORMATIVE_CNR_BY_SEQUENCE[key]
    return NORMATIVE_CNR_BY_SEQUENCE["fallback"]


# =============================================================================
# АВТООПРЕДЕЛЕНИЕ ТИПА ПОСЛЕДОВАТЕЛЬНОСТИ
# =============================================================================

def detect_sequence_type(json_path: Path) -> str:
    """
    Определяет тип NM-MRI последовательности по JSON sidecar.

    Returns: 'tse' | 'mtc_gre' | 'gre' | 'unknown'
    """
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

    # TSE — ищем явные маркеры
    if "TSE" in combined or "TURBO SPIN" in combined:
        return "tse"
    if "SE" in seq and "GR" not in seq:
        return "tse"

    # MTC-GRE — MT-препульс + градиентное эхо
    has_mt = "MT" in opts or "MTC" in combined or "MT_" in combined
    has_gr = "GR" in seq or "GR" in var or "GR" in combined or "FFE" in combined
    if has_mt and has_gr:
        return "mtc_gre"

    # Просто GRE
    if has_gr:
        return "gre"

    return "unknown"


# =============================================================================
# ИЗВЛЕЧЕНИЕ ИЗ DICOM
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
    """DICOM → JSON fallback."""
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
    norm = get_normative(patient.field_strength, sequence)
    expected = norm["left_sn"]["mean"]
    sd = norm["left_sn"]["sd"]

    z = (measured_cnr_pct - expected) / sd if sd > 0 else 0
    pct_dev = 100.0 * (measured_cnr_pct - expected) / expected if expected else 0.0

    if z < -2.0: interp = "abnormal_low"
    elif z < -1.5: interp = "borderline_low"
    elif z > 2.0: interp = "abnormal_high"
    elif z > 1.5: interp = "borderline_high"
    else: interp = "normal"

    return ComparisonResult(
        measured_cnr_pct=measured_cnr_pct,
        expected_cnr_pct=expected,
        z_score=z,
        percent_deviation=pct_dev,
        interpretation=interp,
        age_used=patient.age,
        sex_used=patient.sex,
        field_used=patient.field_strength,
        sequence_used=sequence,
        normative_source=norm["description"],
        warning=norm.get("warning"),
    )