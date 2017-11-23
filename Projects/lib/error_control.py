import numpy as np


# x: message for error control
# P: parity matrix
# n: number of columns
# r: number of bits to divide the message
def hamming(x: np.ndarray, P: np.ndarray, n: np.int, r: np.int) -> np.ndarray:
    # G = np.hstack((np.identity(n - len(P[0]), dtype='uint8'), P))
    G = np.hstack((np.identity(r, dtype='uint8'), P))

    # C = np.logical_xor(m, G)
    return np.dot(x, G) % 2


# y: received message
# P: parity matrix
def correction(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    # Form the H matrix
    H = np.vstack((P, np.identity(len(P[0]))))

    # Calculate the S matrix
    S = np.dot(y, H) % 2

    # If S is == 0 then there's no error in that sub-message
    # Else find the row position where sub-message is equal to S
    # Then flip the bit in the sub-message at row position
    for row in range(len(S)):
        if np.all(S[row] == 0):
            continue

        col = np.argwhere(np.all(H == S[row], axis=1))[0][0]
        y[row, col] = np.logical_not(y[row, col])

    return y[:, 0:-1 * len(P[0])]


def bit_error_rate(x: np.ndarray, y: np.ndarray) -> np.float:
    return np.sum(np.array([x != y])) / len(x)


def ber_after_error_correction(ber_before: np.float, n: np.int) -> np.float:
    return (3 * (n - 1) / 2) * np.power(ber_before, 2)
