# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as la
from scipy.optimize import minimize


class Plan(object):
    def __init__(self, s, q = None):
        self.s = s                                          # Число регрессоров
        if q is None:
            self.q = int(0.5 * s * (s + 1) + 1)                 # Число точек
        else:
            self.q = q
        self.A = np.matrix(np.ndarray((self.s, self.q)))    # Матрица плана: [s,q]
        self.p = np.matrix(np.ndarray((self.q, 1)))
        self.p[:, :] = 1.0 / self.q                         # Веса точек: вектор, или матрица [q,1]

    def partial_inf_matrix(self, i, f, A=None):
        if A is None:
            A = self.A
        return f(A[:, i]) * f(A[:, i]).transpose()

    def inf_matrix(self, f, A=None, p=None):
        if A is None:
            A = self.A
        if p is None:
            p = self.p
        q = self.q
        return sum(self.partial_inf_matrix(i, f, A) * p[i, 0] for i in xrange(q))

    def X(self, f, A=None, p=None):
        log = np.log
        det = la.det
        M   = self.inf_matrix(f, A, p)
        return -log(det(M))

    def mu(self, f):
        M = self.inf_matrix(f)
        return max(np.asscalar((M**(-1) * self.partial_inf_matrix(i, f)).trace()) for i in xrange(self.q))

    def eta(self, f):
        return self.s


def build_plan_dirgrad(f, xi, epsilon=1.0e-3):
    """Синтез оптимального плана с помощью прямой градиентной процедуры. D критерий.
    """
    exit_cond = np.inf
    iter = 0
    while exit_cond > epsilon:
        iter += 1
        detM = la.det(xi.inf_matrix(f))
        mu   = xi.mu(f)
        print 'On iter %d:\n%s\n%s\n\ndet(M(xi)) = %.3lf\nmu(alpha,xi) = %.3lf\n==================' % (
            iter, xi.A, xi.p, detM, mu)

        bnds = ((0.0, 1.0), ) * xi.q                            # Границы для весов p: [0.0 .. 1.0]
        cons = ({'type': 'eq', 'fun': lambda p: sum(p) - 1},)   # Нормированность суммы весов
        res = minimize(lambda p: xi.X(f, p=np.asmatrix(p).reshape((xi.q, 1))),
                       x0=np.squeeze(np.asarray(xi.p)),
                       method='SLSQP',
                       bounds=bnds,
                       constraints=cons)
        p_new = np.matrix(res['x']).reshape((xi.q, 1))

        bnds = ((-5.0, 5.0), ) * xi.s * xi.q                    # Границы для точек плана A: [-5.0 .. 5.0]
        res = minimize(lambda A: xi.X(f, A=np.asmatrix(A).reshape((xi.s, xi.q))),
                       x0=np.squeeze(np.asarray(xi.A)),
                       method='SLSQP',
                       bounds=bnds)
        A_new = np.matrix(res['x']).reshape((xi.s, xi.q))

        exit_cond = sum(np.linalg.norm(A_new[:,i] - xi.A[:,i])**2 + (p_new[i,0] - xi.p[i,0])**2 for i in xrange(xi.q))
        xi.A[:,:] = A_new
        xi.p[:,:] = p_new
    detM = la.det(xi.inf_matrix(f))
    mu   = xi.mu(f)
    print 'Finally:\n%s\n%s\n\ndet(M(xi)) = %.3lf\nmu(alpha,xi) = %.3lf\n==================' % (
        xi.A, xi.p, detM, mu)
    return xi


def main():
    s  = 2
    q  = 5
    f  = lambda alpha: np.matrix([[np.sin(alpha[0, 0])],
                                  [np.cos(alpha[1, 0])]])
    xi = Plan(s, q)
    xi.A[:, :] = [[-5,  5, 5, -5, 1],
                  [ 5, -5, 5, -5, 0]]
    xi = build_plan_dirgrad(f, xi)

    print 'Checking solution for being optimal (less is better): [%.2lf]' % np.abs(xi.mu(f) - xi.eta(f))


if __name__ == '__main__':
    main()
