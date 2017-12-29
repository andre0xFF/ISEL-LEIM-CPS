import numpy as np
from scipy.special import erfc


def signal_power(x: np.ndarray) -> np.float:
    return np.sum(np.power(x, 2) / len(x))


def time_between_errors(rb: np.int, ber: np.float) -> np.float:
    return 1 / (rb * ber)


def ber(x: np.ndarray, y: np.ndarray) -> np.float:
    return np.sum(np.array([x != y])) / (len(x) * len(x[0]))


def ber_hamming(ber_before: np.float, n: np.int) -> np.float:
    return (3 * (n - 1) / 2) * np.power(ber_before, 2)


def ber_manchester(eb, no):
    return (1 / 2) * erfc(np.sqrt(eb / no))


def eb(a, tb):
    return np.power(a, 2) * tb


def snr(px: np.float, py: np.float) -> np.float:
    return px / py


# snr_db == snr_theoric
def snr_db(px: np.float, py: np.float) -> np.float:
    return 10 * np.log10(px / py)


def snr_theoric(r: np.int, p: np.float, vmax: np.float) -> np.float:
    return 6.02 * r + 10 * np.log10(3 * p / np.power(vmax, 2))
