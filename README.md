# Exercise 01 - Signal Frequency Recognition

## Project Overview

This project was created for Exercise 01 in the AI Agents course.

The goal of the project is to generate noisy sine-wave signals and train three neural network models to recognize the frequency of each signal.

The models used in this project are:

- MLP
- RNN
- LSTM

The implementation uses Python and PyTorch.



## Problem Description

A sine wave can be represented as a sequence of numerical samples.

In this exercise, each input contains 10 samples from a noisy sine wave.  
The model receives these 10 samples and tries to predict which frequency class created the signal.

The task is a simple time-series classification problem.

---

## Dataset

The dataset is generated in `src/dataset.py`.

The project uses four fixed frequencies:

- 1 Hz
- 3 Hz
- 5 Hz
- 7 Hz

For each example, the program:

1. Chooses one of the four frequencies.
2. Generates a clean sine wave.
3. Adds random noise to the signal.
4. Uses 10 noisy samples as the input.
5. Saves the correct frequency class as the label.

The context window size is 10 samples.

---

## Models

The models are implemented in `src/models.py`.

### MLP

The MLP receives the 10 samples as one fixed vector.

### RNN

The RNN reads the 10 samples as a sequence and uses a recurrent hidden state.

### LSTM

The LSTM also reads the samples as a sequence, but uses an LSTM layer to learn temporal patterns.

---

## Training

The training code is implemented in `src/train.py`.

The models are trained using:

- Mean Squared Error loss
- Adam optimizer
- One-hot encoded labels

During training, the program prints the loss and accuracy for each model.

---

## Results

All three models trained successfully.

In the final run, MLP, RNN, and LSTM reached `1.0` final accuracy on the generated training dataset.

Even though the final accuracy was the same, the models are different.  
The MLP treats the input as a fixed vector, while RNN and LSTM process the input as a sequence.

---

## Tests

Unit tests are included in `tests/test_dataset.py`.

The tests check:

- The shape of the generated sine signal
- The shape of the dataset
- That the labels are valid

The final test result was:

```text
3 passed
```

---

## AI Agent Usage

An AI coding agent was used during the development process.

The agent helped with:

- Understanding the requirements
- Planning the dataset
- Writing the PyTorch model structure
- Reviewing the code
- Creating documentation

The prompt workflow is documented in:

```text
docs/PROMPTS.md
```

---

## Project Structure

```text
uoh-rl07-ex01/
├── README.md
├── .gitignore
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   ├── TODO.md
│   └── PROMPTS.md
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   └── main.py
└── tests/
    └── test_dataset.py
```

---

## How to Run

Install dependencies:

```bash
pip install numpy pytest torch
```

Run the training:

```bash
python src/train.py
```

Run the tests:

```bash
pytest tests
```

Run a simple model output example:

```bash
python src/main.py
```
