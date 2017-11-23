import numpy as np


def signal_power(x: np.ndarray) -> np.float:
    return np.sum(np.power(x, 2) / len(x))


# r: number of bits
# p: signal power
# v: vmax
def snr_theoric(r: np.int, p: np.float, vmax: np.float) -> np.float:
    return 6.02 * r + 10 * np.log10(3 * p / np.power(vmax, 2))


def snr_pratic(p_signal: np.float, p_quantized: np.float) -> np.float:
    return 10 * np.log10(p_signal / p_quantized)


def time_between_errors(rb: np.int, ber: np.float) -> np.float:
    return 1 / (rb * ber)
