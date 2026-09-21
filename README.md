# QSM + Neuromelanin Pipeline

Python-пайплайн для количественной оценки магнитной восприимчивости (QSM)
и нейромеланина (NM-CNR) из многоэховых МРТ-данных.

## Возможности

### QSM
- Автоопределение серий и переименование in-place
- Чтение TE из JSON sidecar
- Маска мозга: FSL BET / HD-BET / Otsu
- Развёртка фазы: Laplacian, ROMEO, PRELUDE
- Удаление фона: RESHARP, PDF
- Дипольная инверсия: Tikhonov, TKD
- Экспорт в ppm с DC-центрированием

### Neuromelanin
- ANTs SyN + Rigid регистрация атласа KCL
- Автокалибровка латерального сдвига SN
- NM-CNR по KCL-конвенции (mode Crus Cerebri)
- Демография пациента из DICOM
- Автоопределение последовательности (TSE / MTC-GRE)
- Сравнение с нормой (Al Haddad 2023, Chen 2014)
- JSON-отчёт + PNG-монтаж

## Требования

- Python 3.11
- FSL 6.x (bet, flirt, fnirt, applywarp)
- ANTs (через antspyx)

## Установка

```bash
git clone [https://github.com/<username>/qsm-pipeline.git](https://github.com/zensmoke/QSM.git)
cd QSM

# Скачать KCL атлас в папку KCL/ (см. ниже)
# https://github.com/lukevano/KCL_Neuromelanin-MRI
git clone https://github.com/lukevano/KCL_Neuromelanin-MRI.git KCL

# Создать окружение
conda create -n qsm_nm python=3.11 -y
conda activate qsm_nm

# Установить зависимости
pip install -r requirements.txt
```

## Быстрый старт
QSM
```bash
python qsm.py \
  --data-dir /путь/к/data/nifti \
  --output-dir /путь/к/output
```

Neuromelanin
```bash
python neuromelanin/run.py \
  --data-dir /путь/к/data/nifti \
  --dicom-dir /путь/к/data/dicom \
  --output-dir /путь/к/output \
  --kcl-dir /путь/к/KCL \
  --syn
```

## Структура проекта
```text
qsm-pipeline/
├── qsm.py                 ← главный QSM-скрипт
├── config.example.yaml    ← шаблон конфига
├── preprocessing/         ← автоопределение и переименование
├── qsm_io/                ← загрузка данных, маски
├── phase_unwrapping/      ← Laplacian, ROMEO, PRELUDE
├── background_removal/    ← RESHARP, PDF
├── dipole_inversion/      ← Tikhonov, TKD
├── neuromelanin/          ← NM-анализ
├── tools/                 ← утилиты и диагностика
└── utils/                 ← работа с NIfTI
```

Если возникнуть вопросы по работе пайплайна (QSM или NM), можете прописать флаг --help
```bash
#QSM

python qsm.py --help

#NM

python neuromelanin/run.py --help
```





