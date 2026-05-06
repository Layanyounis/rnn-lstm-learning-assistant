# Exercise 01 - Signal Frequency Recognition using MLP, RNN, and LSTM

## 1. Project Overview

This project was created for Exercise 01 in the AI Agents course.

The goal of the project is to generate noisy sine-wave signals and train neural network models to recognize the frequency class of each signal.

The project uses three neural network architectures:

- Fully Connected Neural Network (MLP)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory Network (LSTM)

The implementation is written in Python using PyTorch.

---

## 2. Problem Description

A signal can be represented as a sequence of numerical samples.

In this exercise, each input example is a short window of 10 samples taken from a sine wave with added noise.

The model receives the noisy samples and learns to predict which frequency generated the signal.

This is a time-series learning problem because the input is a sequence of values.

---

## 3. Dataset Generation

The dataset is generated synthetically in `src/dataset.py`.

The project uses four fixed frequencies:

- 1 Hz
- 3 Hz
- 5 Hz
- 7 Hz

For each generated example:

1. A frequency is selected from the four fixed frequencies.
2. A clean sine wave is generated.
3. Random noise is added to the signal.
4. A context window of 10 noisy samples is used as the model input.
5. The label represents the frequency class.

The sine signal is generated using the formula:

```text
sin(2 * pi * frequency * time)
```

Noise is added to make the task more realistic and to prevent the models from learning only perfect clean signals.

---

## 4. Models

The project implements three PyTorch models in `src/models.py`.

### 4.1 MLP Model

The MLP model treats the 10 samples as a fixed input vector.

It uses fully connected layers to map the input samples to one of four frequency classes.

### 4.2 RNN Model

The RNN model reads the 10 samples as a sequence.

It uses a recurrent hidden state to process the signal step by step and learn temporal information.

### 4.3 LSTM Model

The LSTM model also reads the signal as a sequence.

It uses an LSTM layer, which is designed to learn temporal patterns and handle memory more effectively than a basic RNN.

---

## 5. Training

The training process is implemented in `src/train.py`.

The models are trained using:

- PyTorch
- Mean Squared Error Loss (MSE)
- Adam optimizer
- One-hot encoded frequency labels

The MSE loss compares the model output with the correct one-hot label vector.

During training, the script prints:

- Model name
- Epoch number
- Loss value
- Accuracy value
- Final accuracy
- True labels
- Predicted labels

---

## 6. Results

The three models were trained successfully using PyTorch.

In the final run, all three models reached high accuracy on the generated dataset.

The MLP, RNN, and LSTM models all reached `1.0` final accuracy on the training set.

Although all three models reached the same final training accuracy, they are different architectures.

The MLP treats the 10 samples as a fixed vector, while the RNN and LSTM process the samples as a sequence.

This shows that the generated sine-wave dataset was learnable and that the three neural-network architectures were able to recognize the signal frequency.

---

## 7. Unit Tests

The project includes unit tests in `tests/test_dataset.py`.

The tests check:

- The shape of the clean and noisy sine signals
- The shape of the generated dataset
- That all labels are in the valid range

The tests were run using `pytest`.

Final test result:

```text
3 passed
```

---

## 8. AI Agent Usage

This project was developed with the help of an AI coding agent.

The AI agent was used for:

- Understanding the homework requirements
- Designing the dataset
- Implementing the PyTorch MLP, RNN, and LSTM models
- Writing and running unit tests
- Reviewing the documentation
- Fixing documentation inconsistencies

The prompt workflow is documented in:

```text
docs/PROMPTS.md
```

---

## 9. Project Structure

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

## 10. Installation

Install the required dependencies:

```bash
pip install numpy pytest torch
```

---

## 11. How to Run

Run the training script:

```bash
python src/train.py
```

Run the unit tests:

```bash
pytest tests
```

Run the simple model output demo:

```bash
python src/main.py
```

---

## 12. Final Verification

The final verification commands were:

```bash
python src/train.py
pytest tests
```

The training script ran successfully for all three models:

- MLP
- RNN
- LSTM

The unit tests also passed successfully.

---

## 13. Notes

When a requirement was not explicitly defined, a simple and reasonable implementation choice was made.

The selected frequencies are fixed throughout the project:

```text
1 Hz, 3 Hz, 5 Hz, 7 Hz
```

The context window size is fixed at 10 samples.

The project focuses on the core homework requirements: dataset generation, MLP/RNN/LSTM implementation, MSE training, documentation, and unit testing.