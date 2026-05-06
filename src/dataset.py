import numpy as np


FREQUENCIES = [1, 3, 5, 7]


def generate_sine(frequency, num_samples=10, noise_level=0.1):
    t = np.linspace(0, 1, num_samples)

    clean_signal = np.sin(2 * np.pi * frequency * t)

    noise = noise_level * np.random.randn(num_samples)
    noisy_signal = clean_signal + noise

    return clean_signal, noisy_signal


def create_dataset(num_samples=100):
    X = []
    y = []

    for _ in range(num_samples):
        class_index = np.random.randint(0, len(FREQUENCIES))
        frequency = FREQUENCIES[class_index]

        clean_signal, noisy_signal = generate_sine(frequency)

        X.append(noisy_signal)
        y.append(class_index + 1)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = create_dataset(5)

    print("Samples:")
    print(X)

    print("Labels:")
    print(y)