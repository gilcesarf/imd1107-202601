#!/usr/bin/env bash

set -euo pipefail

VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

echo "Verificando interpretador Python..."
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Erro: ${PYTHON_BIN} não foi encontrado fora do ambiente virtual."
    echo "Instale o Python 3.12 ou informe explicitamente o interpretador:"
    echo "PYTHON_BIN=/caminho/para/python3.12 ./recreate-venv.sh"
    exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" --version 2>&1)"
PYTHON_PATH="$(command -v "${PYTHON_BIN}")"

echo "Interpretador: ${PYTHON_PATH}"
echo "Versão       : ${PYTHON_VERSION}"

if [[ "${PYTHON_VERSION}" != Python\ 3.12.* ]]; then
    echo "Erro: este projeto deve utilizar Python 3.12."
    exit 1
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "Erro: arquivo ${REQUIREMENTS_FILE} não encontrado."
    exit 1
fi

echo
echo "Removendo ambiente virtual existente: ${VENV_DIR}"
rm -rf "${VENV_DIR}"

echo "Criando ambiente virtual com ${PYTHON_PATH}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "Ativando ambiente virtual"
source "${VENV_DIR}/bin/activate"

echo "Atualizando pip"
python -m pip install --upgrade pip

echo "Instalando dependências"
python -m pip install -r "${REQUIREMENTS_FILE}"

echo
echo "Validando ambiente criado"
python --version
which python

echo
echo "Ambiente recriado com sucesso."
echo "Para ativá-lo em outro terminal:"
echo "source ${VENV_DIR}/bin/activate"
