import numpy as np


# idx: indexes from quantization
# r: number of bits
def pcm_encode(idx: np.ndarray, r: np.int) -> np.ndarray:
    # new data type to divide an int32 variable into 4 int8 variables
    dt = np.dtype((np.int32, {'f0': (np.uint8, 3), 'f1': (np.uint8, 2), 'f2': (np.uint8, 1), 'f3': (np.uint8, 0)}))

    # convert the vector into new data type
    idx_uint8 = idx.view(dtype=dt)

    # pack an numpy array with the 4 uint8 variables
    idx_uint8 = np.array([idx_uint8['f0'], idx_uint8['f1'], idx_uint8['f2'], idx_uint8['f3']])

    # transpose so we get each number by row
    idx_uint8 = np.transpose(idx_uint8)

    # convert to binary
    idx_bin = np.unpackbits(idx_uint8, axis=1)

    # slice into the desired number of bits
    idx_bin = idx_bin[:, len(idx_bin[0]) - r:len(idx_bin[0])]

    return idx_bin


def pcm_decode(bits: np.ndarray) -> np.ndarray:
    # return bits.dot(np.power(2, len(bits[0] - 1), -1, -1))
    return bits.dot(1 << np.arange(len(bits[0]) - 1, -1, -1))


def gray_encode(idx, r):
    pass


def gray_decode(bits):
    pass
