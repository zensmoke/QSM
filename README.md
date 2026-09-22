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
- Регистрация атласа KCL через ANTs в 3 режимах + мульти-T1
- Автокалибровка латерального сдвига SN по χ-SN в QSM
- NM-CNR по KCL-конвенции (mode Crus Cerebri)
- Robust NM-CNR (slice-wise + ipsilateral) — диагностический
- Демография пациента из DICOM
- Автоопределение последовательности (TSE / MTC-GRE / GRE)
- Сравнение с нормой (KCL, Al Haddad 2023, Chen 2014)
- JSON-отчёт + PNG-монтаж
- Валидация base-метода: Δ<5e-6 против официальных KCL результатов

## Требования

- Python 3.11
- FSL 6.x (`bet` — для маски мозга)
- ANTs (через `antspyx`)
- Опционально: Julia + ROMEO (для развёртки фазы через ROMEO)

## Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/zensmoke/QSM.git QSM
cd QSM

# 2. Скачать атлас KCL в папку KCL/
git clone https://github.com/lukevano/KCL_Neuromelanin-MRI.git KCL

# 3. Создать окружение
conda create -n qsm_nm python=3.11 -y
conda activate qsm_nm

# 4. Установить зависимости
pip install -r requirements.txt
```

## Быстрый старт

### QSM

```bash
python qsm.py \
    --data-dir /путь/к/data/nifti \
    --output-dir /путь/к/output
```

**Результат:** `output/qsm.nii.gz`, `output/brain_mask.nii.gz`, промежуточные
файлы для QC (`unwrapped_phase.nii.gz`, `local_field.nii.gz`).

### Neuromelanin

```bash
python neuromelanin/run.py \
    --data-dir /путь/к/data/nifti \
    --dicom-dir /путь/к/data/dicom \
    --output-dir /путь/к/output \
    --kcl-dir /путь/к/KCL \
    --multi-t1 \
    --force
```

**Результат:** `output/nm_analysis/` — `report.json`, `qc_montage.png`,
маски SN-VTA и Crus.

## Режимы регистрации NM → MNI

Регистрация атласа KCL (MNI-пространство) в пространство NM-MRI —
ключевой этап. Поддерживаются 4 режима.

### 1. Standard (по умолчанию)

```bash
python neuromelanin/run.py \
    --data-dir ... --output-dir ... --kcl-dir ... \
    --t1 T1_sag.nii --syn
```

Цепочка: `NM --Rigid--> T1 --SyN--> MNI`

**Когда использовать:** T1 полный (z-FOV ≥ 160 мм), хорошего качества.

**Минусы:** если NM и T1 сильно отличаются по контрасту, Rigid может
провалиться.

### 2. Intermediate

```bash
python neuromelanin/run.py \
    --data-dir ... --output-dir ... --kcl-dir ... \
    --t1 T1_sag.nii --t1-other T1_tra.nii
```

Цепочка: `NM --Rigid--> T1_other --Rigid--> T1_anchor --SyN--> MNI`

**Когда использовать:** Standard провалился (атлас уехал, CNR < 0,
χ_SN < 0). Хорошо работает, когда NM и T1_other в одной ориентации
(например, оба axial).

### 3. Direct (без T1)

```bash
python neuromelanin/run.py \
    --data-dir ... --output-dir ... --kcl-dir ... \
    --direct-nm-to-mni --direct-method Affine
```

Цепочка: `NM --Affine/Rigid/SyN--> MNI`

**Когда использовать:** T1 нет или плохого качества.

**Шанс успеха:** низкий — NM-MRI имеет частичный FOV (только средний
мозг), ANTs часто не справляется.

### 4. Мульти-T1 (автоперебор)

```bash
python neuromelanin/run.py \
    --data-dir ... --output-dir ... --kcl-dir ... \
    --multi-t1 --force
```

**Что делает:**
1. Находит все `T1_*.nii*` в `--data-dir`.
2. Строит стратегии:
   - `standard_<plane>` — для каждой найденной плоскости
   - `intermediate_<other>_via_<anchor>` — anchor = T1 с наибольшим FOV
   - `direct_Affine`, `direct_Rigid`
3. Прогоняет каждую через `register.py`.
4. Оценивает атлас в QSM: SN χ mean после автокалибровки сдвига.
5. Копирует лучший `atlas_native.nii.gz` в `--output`.

**Требует:** `qsm.nii.gz` (для скоринга).

**Время:** 5–20 минут (зависит от числа стратегий).

**Когда использовать:**
- Есть несколько T1-плоскостей (sag + tra + cor)
- Не уверены, какая T1 лучше
- Standard/Intermediate провалились

**Пример вывода:**

```
Стратегий к прогону: 5
  • standard_sag
  • standard_tra
  • intermediate_tra_via_sag
  • direct_Affine
  • direct_Rigid

СВОДКА
  Стратегия                   score     χ_mean    L/R  z_frac  shift
  standard_tra              +0.9863    +0.0107   0.19    0.56     24   ← лучшая
  standard_sag              -0.2389    +0.0067   0.00    0.05     10
  intermediate_tra_via_sag  -0.6374    +0.0000   0.00    0.18      0
  direct_Affine             ERROR: too few voxels
  direct_Rigid              ERROR: too few voxels
```

### Когда какой режим использовать

| Ситуация | Рекомендация |
|---|---|
| Есть T1_sag (полный) + T1_tra | `--multi-t1` |
| Есть только T1_sag (полный) | `--t1 T1_sag.nii --syn` |
| Есть только T1_tra (обрезан) | `--t1 T1_tra.nii --syn` (проверить результат) |
| Нет T1 | `--direct-nm-to-mni --direct-method Affine` |
| Standard провалился | `--t1 T1_sag.nii --t1-other T1_tra.nii` |
| Ничего не работает | Ручная сегментация в ITK-SNAP |

### Рекомендация для Philips MTC-GRE

Если NM-MRI аксиальный, а есть `T1_tra` (тоже аксиальный) —
**начинайте с `--multi-t1`**. Опыт показывает, что `standard_tra`
(axial→axial Rigid) часто побеждает `standard_sag` (sagittal→axial SyN).

## Структура проекта

```text
qsm-pipeline/
├── qsm.py                          ← главный QSM-скрипт
├── config.example.yaml             ← шаблон конфига
├── requirements.txt
├── preprocessing/                  ← автоопределение и переименование
│   ├── rename_files.py
│   └── auto_detect.py
├── qsm_io/                         ← загрузка данных, маски
│   ├── data_loader.py
│   ├── fsl_bet.py
│   └── hdbet.py
├── phase_unwrapping/               ← Laplacian, ROMEO, PRELUDE
│   ├── unwrapper.py
│   ├── romeo.py
│   └── prelude.py
├── background_removal/             ← RESHARP, PDF
│   ├── pdf.py
│   └── resharp.py
├── dipole_inversion/               ← Tikhonov, TKD
│   └── tkd.py
├── neuromelanin/                   ← NM-анализ
│   ├── run.py                      ← оркестратор
│   ├── register.py                 ← ANTs (3 режима)
│   ├── register_multi.py           ← перебор стратегий
│   ├── extract_masks.py            ← маски + автокалибровка
│   ├── analyze.py                  ← NM-CNR + QSM + демография
│   ├── cnr_robust.py               ← slice-wise + ipsilateral
│   ├── demographics.py             ← нормативы + DICOM
│   └── normative_db.json           ← KCL-нормативы
├── tools/                          ← утилиты и диагностика
│   ├── build_kcl_normative.py      ← batch-сборка нормативов
│   ├── test_on_kcl.py              ← валидация на KCL
│   ├── plot_echoes.py              ← визуализация развёртки фазы
│   └── ...
└── utils/                          ← работа с NIfTI
    └── nifti_utils.py
```

## Утилиты

### Валидация на KCL-датасете

```bash
python tools/test_on_kcl.py
```

Сравнивает base + robust методы против официального
`NM_CNR_SNVTA_Results.xlsx`. Ожидаемый результат:
- base: Δ < 1e-5 (численно эквивалентно KCL)
- robust: Δ < 1e-3

### Batch-сборка нормативов

```bash
python tools/build_kcl_normative.py \
    --kcl-dir ./KCL \
    --out ./neuromelanin/normative_db.json \
    --csv ./KCL/hc_batch.csv
```

Прогоняет все `sub-*` из KCL, собирает `mean ± SD` в `normative_db.json`.
При `n < 20` SD не записывается (используется floor в `demographics.py`).

### Визуализация развёртки фазы

```bash
python tools/plot_echoes.py \
    /путь/к/output/unwrapped_phase.nii.gz \
    -o /путь/к/output/echoes_grid.png
```

## Выходные файлы NM-анализа

```text
output/nm_analysis/
├── report.json                     ← все метрики
├── qc_montage.png                  ← визуальный QC
├── nm_cnr_map.nii.gz               ← voxel-wise CNR
├── SN_VTA_L_native.nii.gz          ← маска SN-L
├── SN_VTA_R_native.nii.gz          ← маска SN-R
├── CrusCerebri_native.nii.gz       ← маска Crus
├── all_masks_combined.nii.gz       ← combined (1=SN-L, 2=SN-R, 3=Crus)
├── sn_shift_calibration.json       ← результаты автокалибровки
└── _reg/                           ← регистрация
    ├── atlas_native.nii.gz
    ├── reg_meta.json
    └── multi_strategies/           ← если --multi-keep-all
```

**Ключевые поля `report.json`:**

| Ключ | Что содержит |
|---|---|
| `patient` | age, sex, ID, B0, manufacturer |
| `registration.best_strategy` | Какая стратегия победила |
| `nm_cnr` | base CNR (mean, L, R, pct) |
| `nm_cnr_robust` | robust CNR + per-slice диагностика |
| `comparison_with_normative` | z-score, интерпретация |
| `qsm_in_sn` | χ_SN (mean, p95, max), χ_Crus |

## Известные ограничения

### L/R асимметрия χ_SN

В атласе KCL маски SN-L и SN-R несимметричны по форме — R-маска
захватывает больше верхнего края SNpc. Это приводит к χ_SN-R ≈ 2× χ_SN-L.

**Влияние:**
- χ_SN-L и χ_SN-R нельзя интерпретировать отдельно
- χ_SN-среднее (+0.038 ppm) — устойчивая метрика
- CNR L/R симметричен — на него атлас не влияет

### SD в `normative_db.json`

При `n < 20` SD не записывается (см. `build_kcl_normative.py`).
В `demographics.get_normative()` подставляется floor `0.10 × mean`
(≈10% от среднего — реалистичный биологический разброс).

После накопления ≥20 HC **вашего** сканера — перезапустите
`build_kcl_normative.py`, и SD появится в JSON.

## Требуется FSL

Пайплайн использует **только** `fsl bet` для маски мозга.
FLIRT/cropped-MNI/fslreorient2std **не используются** (не справились
с LPS-as-RAS конвенцией Philips DICOM).

## Troubleshooting

### Регистрация провалилась (χ_SN < 0, CNR < 0)

Симптомы:
- `atlas_native.nii.gz` уехал в CSF / вне SN
- `χ SN mean < 0` (SN должна быть парамагнитной)
- `CNR < 0` (SN должна быть ярче Crus на MTC-GRE)

Порядок действий:
1. `--multi-t1 --multi-keep-all` — перебрать все стратегии
2. Посмотреть `_reg/multi_strategies/<strategy>/atlas_native.nii.gz`
   визуально (ITK-SNAP + QSM)
3. Если все плохие — ручная сегментация в ITK-SNAP:
   - Открыть `echo-1_part-mag.nii`
   - Нарисовать SN-L (label 1), SN-R (label 2), Crus (label 3)
   - Сохранить как `SN_VTA_L_native.nii.gz` и т.д. в `nm_analysis/`
   - Пропустить регистрацию, запустить сразу `analyze.py`

### `No module named 'neuromelanin'`

Запускать скрипты из корня проекта, не из `tools/`:

```bash
cd /путь/к/QSM
python tools/test_on_kcl.py
```

### `UnboundLocalError: desc`

Устаревшая версия `run.py`. Обновите из репозитория.

### z-score взрывается (±10 и больше)

SD в `normative_db.json` слишком маленький (n<20).
Убедитесь, что `demographics.py` — последней версии (с floor 0.10).

## Ссылки

- KCL атлас: https://github.com/lukevano/KCL_Neuromelanin-MRI
- Al Haddad 2023: JMRI, TSE-нормативы (n=152)
- Chen 2014: MTC-GRE, Philips (n=12)
- He et al. 2021: NeuroImage, iron + NM simultaneously

## Лицензия

MIT License

Copyright (c) 2026 Nikita Myasov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Контакты

- GitHub Issues: https://github.com/zensmoke/QSM/issues