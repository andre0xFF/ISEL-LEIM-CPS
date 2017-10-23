import numpy as np
import lab01
from lab01.lab01 import sawtooth_signal, quantify



def main():
    exercise_01()
    exercise_02()
    exercise_03()
    exercise_04()
    exercise_05()
    exercise_06()


def exercise_01():
    signal = sawtooth_signal()
    vmax = np.max(np.abs(signal))
    r = 3

    mq, idx = quantify(signal, 'midrise', vmax, r)
    test = np.unpackbits(idx, axis=1)
    print('')


def pcm_encode(idx, r):
    pass


def pcm_decode():
    pass


def exercise_02():
    pass


def exercise_03():
    pass


def exercise_04():
    pass


def exercise_05():
    pass


def exercise_06():
    pass


if __name__ == '__main__':
    main()
