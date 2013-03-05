# -*- coding: utf-8 -*-

from numpy import matrix
import numpy as np
from numpy import linalg as la
from scipy.optimize import minimize


class Plan(object):
    def __init__(self, s):
        self.s = s                                          # Число регрессоров
        q = int(0.5 * s * (s + 1) + 1)
        self.q = q                                          # Число точек
        self.A = np.matrix(np.ndarray((self.s, self.q)))    # Матрица плана
        self.p = np.array(np.ndarray((self.q, )))
        self.p[:] = 1.0 / q                              # Веса точек

    def inf_matrix(self, f, A=None, p=None):
        if A is None:
            A = self.A
        if p is None:
            p = self.p
        q = self.q
        return sum(f(A[:, i]) * f(A[:, i]).transpose() * p[i] for i in xrange(q))

    def grad_by_alpha(self, f, A=None, p=None):
        return -np.log(la.det(self.inf_matrix(f, A, p)))

    def grad_by_p(self, f, A=None, p=None):
        return -np.log(la.det(self.inf_matrix(f, A, p)))
        # if A is None:
        #     A = self.A
        # q = self.q
        # return np.array([-(la.inv(self.inf_matrix(f, A, p)) * f(A[:, i]) * f(A[:, i]).transpose()).trace()[0, 0]
        #                  for i in xrange(q)])


def build_plan_dirgrad(f, xi0):
    """Синтез оптимального плана с помощью прямой градиентной процедуры. D критерий.
    """
    xi = xi0

    exit_cond = 1.0e+10
    detM = la.det(xi.inf_matrix(f))                 # Определитель информационной матрицы
    while exit_cond > 1.0e-3:
        bnds = ((0.0, 1.0), ) * xi.q                # Границы для весов p: [0.0 .. 1.0]
        cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1},)   # Нормированность суммы весов
        res = minimize(lambda p: xi.grad_by_p(f, p=p),
                       x0=xi.p,
                       method='SLSQP',
                       bounds=bnds,
                       constraints=cons)
        pass


def main():
    xi = Plan(2)
    xi.A[:, :] = [[1, -1,  1,  3],
                  [1,  2, -2,  0]]
    f    = lambda alpha: np.matrix([[alpha[0, 0]   ],
                                    [alpha[1 ,0]**2]])
    build_plan_dirgrad(f, xi)



if __name__ == '__main__':
    main()

