SHELL = /bin/bash
HOST_PYTHON := python3.11
VENV_NAME = mlpipeline_env
VENV_BIN = ${VENV_NAME}/bin
PYTHON := ${VENV_BIN}/python
PIP := ${VENV_BIN}/pip
PYTEST := ${VENV_BIN}/pytest
FLAKE8 := ${VENV_BIN}/flake8
MYPY := ${VENV_BIN}/mypy
PYLINT := ${VENV_BIN}/pylint
BLACK := ${VENV_BIN}/black
ISORT := ${VENV_BIN}/isort
MAIN_FOLDER = pipeline
TEST_FOLDER = tests

.PHONY: venv style test

# Environment
venv:
	${HOST_PYTHON} -m venv ${VENV_NAME} && \
	${PYTHON} -m pip install --upgrade pip setuptools wheel && \
	${PIP} install -r requirements.txt && \
	${PIP} install pytest flake8 mypy pylint black isort pre-commit && \
	${VENV_BIN}/pre-commit install || true

# Style
style:
	${BLACK} ./${MAIN_FOLDER}/
	${FLAKE8} ./${MAIN_FOLDER}/
	${ISORT} -rc ./${MAIN_FOLDER}/

test:
	${FLAKE8} ./${MAIN_FOLDER}/
	${MYPY} ./${MAIN_FOLDER}/
	${PYTEST} -s --durations=0 ${TEST_FOLDER}/
	${PYLINT} ./${MAIN_FOLDER}/
