# PRD - Signal Frequency Recognition

## Project Goal

The goal of this project is to build a small neural-network system that recognizes the frequency of a sine signal from a short window of samples.

The project is based on Exercise 01 from the AI Agents course.

## Problem

A signal can be represented as a sequence of numerical samples.  
The task is to generate sine waves with noise and train models to predict the correct frequency.

## Dataset Requirements

The dataset contains generated sine signals.

Each sample includes:

- A selected frequency from four known frequencies
- A clean sine signal
- A noisy sine signal
- A context window of 10 samples

## Frequencies

The project uses four known frequencies:

- 1 Hz
- 3 Hz
- 5 Hz
- 7 Hz

## Models

The project includes three models:

- Fully Connected Neural Network (MLP)
- Simple RNN
- Simple LSTM

## Loss Function

The project uses Mean Squared Error (MSE) as the loss function.

## Success Criteria

The project is considered successful if:

- The dataset is generated correctly
- The three models run successfully
- The loss is calculated using MSE
- The training process shows decreasing loss
- Unit tests pass successfully

## Assumptions

When a detail was not explicitly defined, a simple and reasonable implementation choice was made.

The RNN and LSTM models are educational simplified implementations.