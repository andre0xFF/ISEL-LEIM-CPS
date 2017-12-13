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


# adapted filter
# y: message transmitted in the channel
# lambda_: decision value threshold
# p: number of points
def machester_decode(y: np.ndarray, lambda_: np.float, p: np.int=8) -> np.ndarray:
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
    bits = machester_decode(y, lambda_, p)

    elem = 0

    for i in range(len(bits)):
        for j in range(len(bits[i])):
            new_elem = bits[i, j]
            bits[i, j] = np.logical_xor(elem, bits[i, j])
            elem = new_elem

    return bits


def constellation_encode(x: np.ndarray) -> (np.ndarray, np.int):
    # the constellation uses 4 bits and converts them to coordinates
    # the received matrix may not have a column length multiple of 4
    # so we need to add them now and remove them in the decode function
    multiple = len(x[0]) % 4
    new_bits = 0

    if multiple != 0:
        new_bits = 4 - multiple
        x = np.concatenate((x, np.zeros((len(x), new_bits), dtype=np.int8)), axis=1)

    coords = np.zeros((len(x), np.int(len(x[0]) / 2)), dtype=np.int8)

    # based on the constellation codification calculate the signal and
    # the (x, y) coordinate
    def signal_calc(s):
        return -2 * s + 1

    def coord_calc(c):
        return 2 * c + 1

    for i in range(0, len(coords[0]), 2):
        signal = signal_calc(x[:, i * 2:i * 2 + 2])
        coord = coord_calc(x[:, i * 2 + 2:i * 2 + 4])

        coords[:, i:i + 2] = np.transpose(np.array([signal[:, 0] * coord[:, 1], signal[:, 1] * coord[:, 0]]))

    return coords, new_bits


def constellation_decode(y: np.ndarray, rm_bits: np.int) -> np.ndarray:
    bits = np.zeros((len(y), np.int(len(y[0]) * 2)), dtype=np.int8)

    for i in range(0, len(y[0]), 2):
        signal_x = (y[:, i + 0] < 0) * 1
        signal_y = (y[:, i + 1] < 0) * 1

        bit_x = np.array((np.abs(y[:, i + 0]) - 1) / 2, dtype=np.uint8)
        bit_y = np.array((np.abs(y[:, i + 1]) - 1) / 2, dtype=np.uint8)

        j = i * 2
        bits[:, j:j + 4] = np.transpose(np.array([signal_x, signal_y, bit_y, bit_x]))

    if rm_bits != 0:
        return bits[:, 0:-1 * rm_bits]

    return bits


def qam_encode(x: np.ndarray, p: np.int) -> (np.ndarray, np.int):
    coords, new_bits = constellation_encode(x)
    n = np.reshape(np.arange(p), (1, p))
    symbols = np.zeros((len(coords), np.int(len(coords[0]) / 2 * p)), dtype=np.float32)

    for i in range(0, len(coords[0]), 2):
        col1 = np.reshape(coords[:, i + 0], (len(coords), 1))
        col2 = np.reshape(coords[:, i + 1], (len(coords), 1))
        j = np.int(p / 2 * i)

        symbols[:, j:j + p] = col1 * np.sqrt(2 / p) * np.cos(2 * np.pi * n * 1 / p) +\
                              col2 * np.sqrt(2 / p) * np.sin(2 * np.pi * n * 1 / p)

    return symbols, new_bits


def qam_decode(y: np.ndarray, p: np.int, rm_bits: np.int) -> np.ndarray:
    n = np.reshape(np.arange(p), (1, p))
    c1 = np.sqrt(2 / p) * np.cos(2 * np.pi * n * 1 / p)
    c2 = np.sqrt(2 / p) * np.sin(2 * np.pi * n * 1 / p)
    coords = np.zeros((len(y), np.int(len(y[0]) * 2 / p)))

    def cell_round(c):
        return 2 * np.ceil(c / 2) - 1

    for i in range(0, len(coords[0]), 2):
        j = np.int(p / 2 * i)
        bits = y[:, j:j + p]

        phi_x = np.sum(bits * c1, axis=1)
        phi_y = np.sum(bits * c2, axis=1)
        phi_x = cell_round(phi_x)
        phi_y = cell_round(phi_y)

        coords[:, i + 0] = phi_x
        coords[:, i + 1] = phi_y

    return constellation_decode(coords, rm_bits)
