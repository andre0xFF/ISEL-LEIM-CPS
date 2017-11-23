import numpy as np


# x: signal (from hamming error control)
# a: amplitude
# p: number of points
def line_code_manchester(x: np.ndarray, a: np.int, p: np.int) -> np.ndarray:
    lines = np.zeros(shape=(len(x), len(x[0]) * p), dtype=np.int32)

    bit = lambda bit, n: np.hstack(((-1) * a * np.ones(n), a * np.ones(n))) if bit \
        else np.hstack((a * np.ones(n), (-1) * a * np.ones(n)))

    for i in range(len(x)):
        for j in range(len(x[i])):
            lines[i, j * p:j * p + p] = bit(x[i, j], np.int(p / 2))

    return lines


def adapted_filter(y: np.ndarray, lambda_: np.float, p: np.int=8) -> np.ndarray:
    bits = np.zeros(shape=(len(y), np.int(len(y[0]) / p)), dtype=np.int8)
    b = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    filter = lambda points: np.sum(points * b)

    for i in range(len(y)):
        for j in range(0, len(y[i]), p):
            correlation = filter(y[i, j:j + p])
            col = np.int(j / 8)
            bits[i, col] = 1 if correlation >= lambda_ else 0

    return bits
