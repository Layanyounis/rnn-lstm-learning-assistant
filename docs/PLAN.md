# PLAN - Signal Frequency Recognition

## Step 1 - Project Structure

Create the project folders:

- docs
- src
- tests

Create the required files:

- README.md
- docs/PRD.md
- docs/PLAN.md
- docs/TODO.md
- src/dataset.py
- src/models.py
- src/train.py
- src/main.py
- tests/test_dataset.py

## Step 2 - Dataset Generation

Implement a dataset generator that creates sine waves.

Each sample includes:

- 10 noisy sine-wave samples
- A frequency label
- A clean sine wave for reference

## Step 3 - Model Implementation

Implement three neural network models:

- MLP
- RNN
- LSTM

## Step 4 - Training

Train each model using the generated dataset.

Use Mean Squared Error (MSE) as the loss function.

## Step 5 - Evaluation

Print:

- Training loss
- True labels
- Predicted labels

Compare the behavior of the three models.

## Step 6 - Unit Tests

Create tests for:

- sine generation shape
- dataset shape
- valid labels

## Step 7 - Documentation

Write a README file as a lab report.

Document:

- project goal
- dataset
- models
- loss function
- results
- how to run