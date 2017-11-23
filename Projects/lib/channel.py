import numpy as np


# x: message from error control
def send_with_binomial_noise(x: np.ndarray, ber: np.float) -> np.ndarray:
    vector = matrix_to_vector(x)

    # Generate error/noise to simulate channel communication
    e = np.random.binomial(1, ber, len(vector))
    y = (vector + e) % 2

    return vector_to_matrix(y, len(x[0]))


# x: code line from digital modulation
def send_with_awgn(x: np.ndarray, sigma: np.float) -> np.ndarray:
    vector = matrix_to_vector(x)

    # Generate error/noise to simulate channel communication
    y = vector + sigma * np.random.randn(len(vector))

    return vector_to_matrix(y, len(x[0]))


def matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    return np.ndarray.flatten(matrix)


# n: number of columns
def vector_to_matrix(vector: np.ndarray, n: np.int) -> np.ndarray:
    row = np.int(len(vector) / n)
    col = n
    return np.reshape(vector, (row, col))


def ber():
    # TODO
    pass


def snr():
    # TODO
    pass
