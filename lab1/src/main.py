# -*- coding: utf-8 -*-

from numpy import matrix
import numpy as np


class Plan(object):
    def __init__(self, s):
        self.s = s                                          # Число регрессоров
        q = int(0.5 * s * (s + 1) + 1)
        self.q = q                                          # Число точек
        self.A = np.matrix(np.ndarray((self.s, self.q)))    # Матрица плана
        self.p = np.array([1.0 / q] * q)                    # Веса точек

    def inf_matrix(self):
        A = self.A
        p = self.p
        q = self.q
        return sum(A[:, i] * A[:, i].transpose() * p[i] for i in xrange(0, q))

    def grad_by_alpha(self):
        return -np.log(np.linalg.det(self.inf_matrix()))

    def grad_by_p(self, f):
        return


def main():
    xi = Plan(2)
    xi.A[:, :] = [[1, -1,  1,  3],
                  [1,  2, -2,  0]]
    f    = lambda alpha: np.matrix([[alpha[0]   ],
                                    [alpha[1]**2]])
    dMdA = [lambda alpha: np.matrix([[2*alpha[0],  alpha[1]**2],
                                     [alpha[1]**2, 0          ]]),
            lambda alpha: np.matrix([[0,                   2*alpha[0]*alpha[1]],
                                     [2*alpha[0]*alpha[1], 4*alpha[1]**3]])]




if __name__ == '__main__':
    main()

