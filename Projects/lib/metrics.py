import numpy as np


def signal_power(x: np.ndarray) -> np.float:
    return np.sum(np.power(x, 2) / len(x))


def time_between_errors(rb: np.int, ber: np.float) -> np.float:
    return 1 / (rb * ber)


def ber(x: np.ndarray, y: np.ndarray) -> np.float:
    return np.sum(np.array([x != y])) / len(x)


def ber_after_error_correction(ber_before: np.float, n: np.int) -> np.float:
    return (3 * (n - 1) / 2) * np.power(ber_before, 2)


def snr_reception(px, p_reception):
    return px / p_reception
