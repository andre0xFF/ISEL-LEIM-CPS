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
    # form the H matrix
    H = np.vstack((P, np.identity(len(P[0]))))

    # calculate the S matrix
    S = np.dot(y, H) % 2

    # if S is == 0 then there's no error in that sub-message
    # else find the row position where sub-message is equal to S
    # then flip the bit in the sub-message at row position
    for row in range(len(S)):
        if np.all(S[row] == 0):
            continue

        col = np.argwhere(np.all(H == S[row], axis=1))[0][0]
        y[row, col] = np.logical_not(y[row, col])

    return y[:, 0:-1 * len(P[0])]
