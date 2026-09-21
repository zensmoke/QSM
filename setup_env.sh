#!/bin/bash
# setup_env.sh — создание единого окружения qsm_nm
set -e

ENV_NAME="qsm_nm"
PY_VERSION="3.11"

echo "═══════════════════════════════════════════════════════════"
echo "  Создание окружения ${ENV_NAME} (Python ${PY_VERSION})"
echo "═══════════════════════════════════════════════════════════"

# 1. Создать окружение
echo ""
echo "▶ Шаг 1/6: Создание окружения..."
conda create -n ${ENV_NAME} python=${PY_VERSION} -y

# 2. Активировать
echo ""
echo "▶ Шаг 2/6: Активация..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# 3. Основные научные через conda
echo ""
echo "▶ Шаг 3/6: Установка научных пакетов (conda-forge)..."
conda install -c conda-forge -y \
    numpy=1.26 \
    scipy=1.15 \
    pandas \
    matplotlib \
    scikit-image \
    scikit-learn \
    statsmodels

# 4. pip-пакеты
echo ""
echo "▶ Шаг 4/6: Установка pip-пакетов..."
pip install --upgrade pip
pip install \
    nibabel>=5.0 \
    pydicom>=2.4 \
    SimpleITK>=2.3 \
    openpyxl>=3.1 \
    PyYAML>=6.0 \
    tqdm>=4.65

# 5. ANTs
echo ""
echo "▶ Шаг 5/6: Установка antspyx (может занять 5-10 мин)..."
if ! pip install antspyx==0.5.4; then
    echo "  ⚠ pip не справился, ставим через conda..."
    conda install -c conda-forge ants -y
    pip install antspyx --no-cache-dir
fi

# 6. KCL-зависимости
echo ""
echo "▶ Шаг 6/6: Установка KCL-зависимостей (nipype и др.)..."
pip install \
    nipype>=1.8.6 \
    traits \
    prov \
    rdflib \
    networkx \
    filelock \
    isodate \
    looseversion \
    etelemetry \
    simplejson

# Проверка
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Проверка окружения"
echo "═══════════════════════════════════════════════════════════"
python -c "
import numpy, scipy, nibabel, skimage, yaml, tqdm
import pydicom, SimpleITK, ants, matplotlib, pandas, openpyxl
import nipype
from nipype.interfaces.fsl import RobustFOV
from nipype.interfaces.base import CommandLine
from nipype.interfaces.ants import Registration
print('✅ Все зависимости установлены')
print(f'  Python:     {__import__(\"sys\").version.split()[0]}')
print(f'  numpy:      {numpy.__version__}')
print(f'  scipy:      {scipy.__version__}')
print(f'  nibabel:    {nibabel.__version__}')
print(f'  SimpleITK:  {SimpleITK.__version__}')
print(f'  antspyx:    {ants.__version__}')
print(f'  nipype:     {nipype.__version__}')
print(f'  pydicom:    {pydicom.__version__}')
"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Окружение ${ENV_NAME} готово!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Активация:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "  Запуск QSM-пайплайна:"
echo "    cd /Users/nikitamyasov/PycharmProjects/QSM"
echo "    python main.py config.yaml"
echo ""
echo "  Запуск NM-анализа:"
echo "    python neuromelanin/run.py --syn"
echo ""
