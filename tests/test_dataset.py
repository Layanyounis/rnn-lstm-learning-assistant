import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import create_dataset, generate_sine


def test_generate_sine_shapes():
    clean, noisy = generate_sine(frequency=2, num_samples=10)

    assert clean.shape == (10,)
    assert noisy.shape == (10,)


def test_create_dataset_shapes():
    X, y = create_dataset(num_samples=20)

    assert X.shape == (20, 10)
    assert y.shape == (20,)


def test_labels_are_valid():
    X, y = create_dataset(num_samples=50)

    assert np.all(y >= 1)
    assert np.all(y <= 4)