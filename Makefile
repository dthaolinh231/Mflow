SHELL = /bin/bash
HOST_PYTHON := python3
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
	${PYTHON} -m pip install pip setuptools wheel && \
	${PYTHON} -m pip install -e .[dev] && \
	${VENV_BIN}/pre-commit install

# Style
style:
	${BLACK} ./${MAIN_FOLDER}/
	${FLAKE8} ./${MAIN_FOLDER}/
	${ISORT} -rc ./${MAIN_FOLDER}/

test:
	python -m flake8 ./${MAIN_FOLDER}/
	python -m mypy ./${MAIN_FOLDER}/
	python -m pytest -s --durations=0 ${TEST_FOLDER}/
	python -m pylint ./${MAIN_FOLDER}/
