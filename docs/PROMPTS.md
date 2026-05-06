# AI-Agent Workflow — Prompts Log

This file documents the prompts used to guide the AI agent through the development of this project.

---

## 1. Requirement Analysis

> I am working on Exercise 01 for an AI Agents course. The project must generate a noisy sine-wave dataset, use four fixed frequencies (1, 3, 5, 7 Hz), apply a context window of 10 samples, implement MLP, RNN, and LSTM models in PyTorch, train with MSE loss, include unit tests, and include README.md, docs/PRD.md, docs/PLAN.md, and docs/TODO.md. Please analyse the requirements and produce a structured plan before writing any code.

---

## 2. Dataset Design

> Implement the dataset module in src/dataset.py. It must: define FREQUENCIES = [1, 3, 5, 7], generate a noisy sine wave with a configurable noise level, return windows of exactly 10 samples, and expose a create_dataset() function that returns NumPy arrays X and y suitable for PyTorch training. Do not implement models or training yet.

---

## 3. PyTorch Model Implementation

> Implement three PyTorch models in src/models.py: an MLP that takes a flat 10-sample input, an RNN that processes the 10 samples as a sequence, and an LSTM that does the same. All three models must output a 4-dimensional vector (one logit per frequency class). Use hidden size 32 throughout.

---

## 4. Training

> Implement the training loop in src/train.py. Load the dataset from src/dataset.py, convert labels to one-hot encoding, and train all three models (MLP, RNN, LSTM) using MSE loss and the Adam optimiser. Print the loss every 30 epochs and report final accuracy for each model.

---

## 5. Testing

> Write unit tests in tests/test_dataset.py using pytest. Cover: (1) the shape of the output of generate_sine(), (2) the shape of X and y returned by create_dataset(), and (3) that all labels in y are in the valid range [1, 4]. Do not test models or training in this file.

---

## 6. Documentation

> Create the following documentation files. README.md: project overview, dataset description, model descriptions, loss function, how to run, and example results. docs/PRD.md: project goal, problem statement, dataset requirements, frequency list (1, 3, 5, 7 Hz), models, loss function, and success criteria. docs/PLAN.md: step-by-step implementation plan. docs/TODO.md: checklist of all tasks with completion status.

---

## 7. Final Review

> Please review my project without changing files yet. Check the project structure and tell me: what files exist, whether the implementation matches the requirements (noisy sine dataset, frequencies 1/3/5/7, context window 10, MLP/RNN/LSTM, MSE loss, unit tests, required docs), and whether anything important is missing or incorrect.
