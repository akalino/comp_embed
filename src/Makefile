## Define Python Environment
PYTHON_INTERPRETER = python3

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Make Dataset
build-data:
	$(PYTHON_INTERPRETER) data/raw/get_data.py

build-training: build-data
	mkdir data/processed data/processed/testing data/processed/training
	$(PYTHON_INTERPRETER) preprocessing.py

## Train model
train-model: build-training
	$(PYTHON_INTERPRETER) train_conv.py