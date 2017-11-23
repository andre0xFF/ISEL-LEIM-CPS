#!/usr/bin/env python3.6
import lib.modulation as m
import lib.channel as c
import numpy as np


def main():
    exercise_02()
    exercise_03()
    exercise_04()
    exercise_05()
    exercise_06()
    exercise_07()
    ex()


def ex():
    t = np.array([[1, 1, 1, 0], [0, 1, 1, 1]])
    l = m.line_code_manchester(t, 1, 8)
    a = m.adapted_filter(l, 0.5, 8)

    vector = c.matrix_to_vector(t)
    matrix = c.vector_to_matrix(vector, len(t[0]))
    print('')


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


def exercise_07():
    pass


if __name__ == '__main__':
    main()