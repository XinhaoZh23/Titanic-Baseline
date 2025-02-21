# Titanic Survival Prediction with PyTorch

This project implements a neural network model using PyTorch to predict survival on the Titanic.

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download the Titanic dataset:
   - Go to https://www.kaggle.com/c/titanic/data
   - Download `train.csv`
   - Place `train.csv` in the project root directory

## Running the Model

Execute the main script:
```bash
python titanic_baseline.py
```

## Features

- Data preprocessing including handling missing values
- Feature standardization
- Train/validation split
- Simple neural network architecture with dropout
- Training progress monitoring
- Model evaluation

## Model Architecture

- Input layer (3 features): Age, Pclass, Sex
- Hidden layer 1: 16 neurons with ReLU activation
- Hidden layer 2: 8 neurons with ReLU activation
- Output layer: 1 neuron with Sigmoid activation
- Dropout layers (0.3) for regularization
