import numpy as np


# x: signal (from hamming error control)
# a: amplitude
# p: number of points
def manchester_enconde(x: np.ndarray, a: np.int, p: np.int=8) -> np.ndarray:
    lines = np.zeros(shape=(len(x), len(x[0]) * p), dtype=np.int32)

    bit = lambda bit, n: np.hstack(((-1) * a * np.ones(n), a * np.ones(n))) if bit \
        else np.hstack((a * np.ones(n), (-1) * a * np.ones(n)))

    for i in range(len(x)):
        for j in range(len(x[i])):
            lines[i, j * p:j * p + p] = bit(x[i, j], np.int(p / 2))

    return lines


# y: message transmitted in the channel
# lambda_: decision value threshold
# p: number of points
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


# x: signal (from hamming error control)
# a: amplitude
# p: number of points
def differential_manchester_encode(x: np.ndarray, a: np.int, p: np.int=8) -> np.ndarray:
    x = x.copy()
    lines = np.zeros(shape=x.shape, dtype=np.int8)

    elem = 0

    for i in range(len(x)):
        for j in range(len(x[i])):
            lines[i, j] = np.logical_xor(elem, x[i, j])
            elem = lines[i, j]

    return manchester_enconde(lines, a, p)


# y: message transmitted in the channel
# lambda_: decision value threshold
# p: number of points
def differential_manchester_decode(y: np.ndarray, lambda_: np.float, p: np.int=8) -> np.ndarray:
    bits = adapted_filter(y, lambda_, p)

    elem = 0

    for i in range(len(bits)):
        for j in range(len(bits[i])):
            new_elem = bits[i, j]
            bits[i, j] = np.logical_xor(elem, bits[i, j])
            elem = new_elem

    return bits
