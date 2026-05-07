#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# setup.sh — bootstrap del proyecto nyc_taxi_eda
# ----------------------------------------------------------------------------
# Crea un venv llamado 'fiuba' dentro de la carpeta del proyecto, instala
# dependencias y registra un kernel Jupyter para que el notebook use SIEMPRE
# el Python del venv (no el Python del sistema/macOS).
#
# Uso:
#   chmod +x setup.sh
#   ./setup.sh
#
# Requisitos: Python 3.10+ instalado en el sistema.
# Compatible: macOS y Linux.
# ----------------------------------------------------------------------------
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PY_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="fiuba"
KERNEL_NAME="nyc_taxi_eda_fiuba"
KERNEL_DISPLAY="Python (fiuba)"

echo ">> Carpeta del proyecto : $PROJECT_DIR"
echo ">> Python del sistema   : $($PY_BIN --version 2>&1)"

# 1. Crear venv si no existe
if [[ ! -d "$VENV_DIR" ]]; then
    echo ">> Creando venv en ./$VENV_DIR ..."
    "$PY_BIN" -m venv "$VENV_DIR"
else
    echo ">> ./$VENV_DIR ya existe, se reutiliza."
fi

# 2. Activar venv y actualizar pip
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo ">> Python del venv      : $(python --version)"
echo ">> Pip del venv         : $(pip --version)"

python -m pip install --upgrade pip --quiet

# 3. Instalar dependencias
echo ">> Instalando dependencias (puede tardar ~1 min la primera vez)..."
pip install -r requirements.txt --quiet

# 4. Registrar kernel en Jupyter (se asocia al venv 'fiuba')
python -m ipykernel install --user --name "$KERNEL_NAME" \
    --display-name "$KERNEL_DISPLAY" >/dev/null

echo ""
echo "=================================================================="
echo "  Setup completado."
echo "  - Activar venv      : source $VENV_DIR/bin/activate"
echo "  - Correr el EDA     : python eda_taxis.py"
echo "  - Abrir el notebook : jupyter lab eda_taxis.ipynb"
echo "                        (selecciona el kernel '$KERNEL_DISPLAY')"
echo "  - Desactivar venv   : deactivate"
echo "=================================================================="
