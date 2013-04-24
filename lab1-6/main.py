# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as la
from scipy.optimize import minimize

import pylab as pl
import IMF
from IMF import IMFSolver

class Plan(object):
    def __init__(self, N, s, q=None):
        self.N = N                                          # Число варьируемых переменных
        self.s = s                                          # Число параметров
        if q is None:
            self.q = int(0.5 * s * (s + 1) + 1)                 # Число точек
        else:
            self.q = q
        self.A = np.matrix(np.ndarray((self.s, self.q)))    # Матрица плана: [s,q]
        self.p = np.matrix(np.ndarray((self.q, 1)))
        self.p[:, :] = 1.0 / self.q                         # Веса точек: вектор, или матрица [q,1]

    def set_bounds(self, bounds):
        self.bnds = bounds

    def _remove_ith_point(self, i):
        p = self.p[i, 0]
        self.p = np.delete(self.p, i, 0)
        self.p[:, :] /= sum(self.p[:, :])
        self.A = np.delete(self.A, i, 1)
        self.q -= 1

    def _add_point(self, a, p):
        self.A = np.append(self.A, a, 1)
        self.p[:, :] *= 1 - p
        self.p = np.append(self.p, [[p]], 0)
        self.q += 1

    def move_point(self, i, a):
        self.A[:, i] = a

    def reduce_plan(self):
        epsilon = 1.0e-2
        i = 0
        while i < self.q:
            if self.p[i, 0] < epsilon:
                self._remove_ith_point(i)
            else:
                i += 1
        i = 0
        while i < self.q:
            j = i + 1
            while j < self.q:
                if la.norm(self.A[:, i] - self.A[:, j]) < epsilon:
                    self.p[i, 0] += self.p[j, 0]
                    self._remove_ith_point(j)
                else:
                    j += 1
            i += 1

    def set_inf_matrix_solver(self, solver):
        self.solver = solver

    def partial_inf_matrix(self, i, A=None):
        if A is None:
            A = self.A
        return self.solver(A[:, i])

    def inf_matrix(self, A=None, p=None):
        if A is None:
            A = self.A
        if p is None:
            p = self.p
        q = self.q
        return sum(self.partial_inf_matrix(i, A) * p[i, 0] for i in xrange(q))

    def X(self, A=None, p=None):
        log = np.log
        det = la.det
        M   = self.inf_matrix(A, p)
        return -log(det(M))

    def mu(self):
        M = self.inf_matrix()
        return max(np.asscalar((M**(-1) * self.partial_inf_matrix(i)).trace()) for i in xrange(self.q))

    def eta(self):
        return self.s

    def jac_p(self, p):
        return np.array([-np.trace(self.inf_matrix(p=p)**(-1) * self.partial_inf_matrix(i)) for i in xrange(self.q)])

    def jac_A(self, dM):
        M = self.inf_matrix()
        return np.matrix([[-np.trace(M**(-1) * self.p[i, 0] * dM(self.A[:, i])[j]) for i in xrange(self.q)] for j in xrange(self.s)])

    def jac_mu(self, dM, A):
        M = self.inf_matrix()
        return np.matrix([[-np.trace(M**(-1) * dM(A)[j])] for j in xrange(self.s)])

    def __str__(self):
        return '%s\n%s' % (self.A, self.p)


def build_plan_dirgrad(xi, dM, epsilon=1.0e-6):
    """Синтез оптимального плана с помощью прямой градиентной процедуры. D критерий.
    """
    exit_cond = np.inf
    iter = 0
    while exit_cond > epsilon:
        iter += 1
        detM = np.log(la.det(xi.inf_matrix()))
        mu   = xi.mu()
        print 'On iter %d:\n%s\n\nlog(det(M(xi))) = %.3lf\nmu(alpha,xi) = %.3lf\n==================' % (
            iter, xi, detM, mu)

        bnds = reduce(lambda a, b: a + b, [(xi.bnds[i], ) * xi.q for i in xrange(xi.s)])     # Границы для точек плана
        res = minimize(lambda A: xi.X(A=np.asmatrix(A).reshape((xi.s, xi.q))),
                       # jac=lambda A: np.squeeze(np.asarray(xi.jac_A(dM).reshape((1, xi.s * xi.q)))),
                       x0=np.squeeze(np.asarray(xi.A.reshape((1, xi.s * xi.q)))),
                       method='SLSQP',
                       bounds=bnds)
        A_new = np.matrix(res['x']).reshape((xi.s, xi.q))
        xi.A[:, :] = A_new

        bnds = ((0.0, 1.0), ) * xi.q                            # Границы для весов p: [0.0 .. 1.0]
        cons = ({'type': 'eq', 'fun': lambda p: sum(p) - 1},)   # Нормированность суммы весов
        res = minimize(lambda p: xi.X(p=np.asmatrix(p).reshape((xi.q, 1))),
                       # jac=lambda p: xi.jac_p(p=np.asmatrix(p).reshape((xi.q, 1))),
                       x0=np.squeeze(np.asarray(xi.p)),
                       method='SLSQP',
                       bounds=bnds,
                       constraints=cons)
        p_new = np.matrix(res['x']).reshape((xi.q, 1))

        exit_cond = sum(np.linalg.norm(A_new[:, i] - xi.A[:, i])**2 + (p_new[i, 0] - xi.p[i, 0])**2 for i in xrange(xi.q))
        xi.p[:, :] = p_new
    detM = np.log(la.det(xi.inf_matrix()))
    mu   = xi.mu()
    print 'Finally:\n%s\n\nlog(det(M(xi))) = %.3lf\nmu(alpha,xi) = %.3lf\n==================' % (
        xi, detM, mu)

    xi.reduce_plan()
    detM = np.log(la.det(xi.inf_matrix()))
    mu   = xi.mu()
    print 'After reduction:\n%s\n\nlog(det(M(xi))) = %.3lf\nmu(alpha,xi) = %.3lf\n==================' % (
        xi, detM, mu)
    return xi


def build_plan_dualgrad(xi, dM, epsilon=1.0e-6):
    """Синтез оптимального плана с помощью двойственной градиентной процедуры. D критерий.
    """
    exit_cond = np.inf
    iter = 0

    buf = xi.A[:, 0]
    while exit_cond > epsilon:
        iter += 1
        detM = np.log(la.det(xi.inf_matrix()))
        mu   = xi.mu()
        print 'On iter %d:\n%s\n\nlog(det(M(xi))) = %.3lf\nmu(alpha,xi) = %.3lf\n==================' % (
            iter, xi, detM, mu)

        a = np.matrix([[np.random.uniform(b[0], b[1])] for b in xi.bnds])
        xi._add_point(a, 1.0 / (xi.q + 1.0))

        def variate_point(xi, i, a):
            xi.A[:, i] = a
            return xi.mu()
            return -la.det(xi.inf_matrix())

        bnds = xi.bnds
        res = minimize(lambda A: variate_point(xi, -1, np.asmatrix(A).reshape((xi.s, 1))),
                       # jac=lambda A: np.squeeze(np.asarray(xi.jac_mu(dM, np.asmatrix(A).reshape((xi.s, 1))).reshape((1, xi.s)))),
                       x0=np.squeeze(np.asarray(xi.A[:, -1].reshape((1, xi.s)))),
                       method='SLSQP',
                       bounds=bnds)
        xi.A[:, -1] = np.matrix(res['x']).reshape((xi.s, 1))

        print 'Found:\n%s\n' % res['x']

        def variate_weight(xi, i, p):
            xi.p[:, :] *= (1 - p) / (sum(xi.p[:, :]) - xi.p[i, 0])
            xi.p[i, 0] = p
            return -la.det(xi.inf_matrix())

        bnds = ((0.0, 1.0), )
        res = minimize(lambda p: variate_weight(xi, -1, p),
                       x0=(0.5,),
                       method='SLSQP',
                       bounds=bnds)
        xi.p[-1, :] = np.matrix(res['x'])

        print 'With weight:\n%s\n' % xi.p[-1, 0]
        xi.reduce_plan()

        exit_cond = xi.mu() - xi.eta()
    detM = np.log(la.det(xi.inf_matrix()))
    mu   = xi.mu()
    print 'Finally:\n%s\n\nlog(det(M(xi))) = %.3lf\nmu(alpha,xi) = %.3lf\n==================' % (
        xi, detM, mu)

    bnds = ((0.0, 1.0), ) * xi.q                            # Границы для весов p: [0.0 .. 1.0]
    cons = ({'type': 'eq', 'fun': lambda p: sum(p) - 1},)   # Нормированность суммы весов
    res = minimize(lambda p: xi.X(f, p=np.asmatrix(p).reshape((xi.q, 1))),
                   x0=np.squeeze(np.asarray(xi.p)),
                   method='SLSQP',
                   bounds=bnds,
                   constraints=cons)
    xi.p[:, :] = np.matrix(res['x']).reshape((xi.q, 1))

    return xi


def build_plan_dirscan(xi0):
    base = np.log(la.det(xi0.inf_matrix()))
    print 'Base plan: \n%s\nlog(det(M))=%.3lf\n=================\n' % (xi0, base)

    xi = Plan(xi0.s, xi0.q + 1)
    xi.set_inf_matrix_solver(xi0.solver)
    xi.set_bounds(xi0.bnds)
    xi.A[:, :-1] = xi0.A[:, :]
    xi.A[:,  -1] = 0
    xi.p[:-1, :] = xi0.p * (float(xi0.q) / (xi0.q + 1))
    xi.p[ -1, :] = 1.0 / (xi0.q + 1)

    def variate_point(xi, i, x, y):
        xi.A[0, i] = x
        xi.A[1, i] = y
        return la.det(xi.inf_matrix())

    func = lambda x, y: variate_point(xi, -1, x, y) - base
    dx = (xi.bnds[0][1] - xi.bnds[1][0]) / 50
    dy = (xi.bnds[1][1] - xi.bnds[1][0]) / 50
    x = np.arange(xi.bnds[0][0], xi.bnds[0][1] + dx, dx)
    y = np.arange(xi.bnds[1][0], xi.bnds[1][1] + dy, dy)
    X, Y = pl.meshgrid(x, y)
    Z = np.array([[func(xv, yv) for xv in x] for yv in y])
    pl.pcolor(X, Y, Z)
    pl.xlim(xi.bnds[0])
    pl.ylim(xi.bnds[1])
    pl.colorbar()
    pl.title('Improving plan from base det(M) = %.3lf by adding a point' % base)

    ij = np.unravel_index(Z.argmax(), Z.shape)
    xi.A[0, -1] = X.item(ij)
    xi.A[1, -1] = Y.item(ij)
    print 'Found plan: \n%s\nlog(det(M))=%.3lf\n=================\n' % (xi, np.log(la.det(xi.inf_matrix())))

    pl.show()
    return xi


def main():
    N = 30       # Число срезов во времени
    n = 2
    r = 1
    p = 2
    m = 1
    s = 4

    q = 4      # Число точек плана

    f = lambda alpha: np.matrix([[np.sin(alpha[0,0])],
                                 [np.cos(alpha[1,0])]])
    dM = lambda alpha: [
        np.matrix([[2 * np.sin(alpha[0,0]) * np.cos(alpha[0,0]), np.cos(alpha[0,0]) * np.cos(alpha[1,0])     ],
                   [    np.cos(alpha[0,0]) * np.cos(alpha[1,0]), 0              ]]),

        np.matrix([[             0,                              -np.sin(alpha[0,0]) * np.sin(alpha[1,0])      ],
                   [    -np.sin(alpha[0,0]) * np.sin(alpha[1,0]),-2 * np.sin(alpha[1,0]) * np.cos(alpha[1,0]) ]]),
        ]


##    f = lambda alpha: np.matrix([[(alpha[0,0])],
##                                 [(alpha[1,0])]])
##    dM = lambda alpha: [
##        np.matrix([[2 * alpha[0,0], alpha[1,0]     ],
##                   [    alpha[1,0], 0              ]]),
##
##        np.matrix([[             0, alpha[0,0]     ],
##                   [    alpha[0,0], 2 * alpha[1,0] ]]),
##        ]

    def build_solver(alpha):
        solver = IMFSolver(n=n, r=r, p=p, m=m, s=s, N=N)

        theta = [0.58, 0.57, 0.52, -0.01]

        solver.set_Phi([[theta[2], 1.0], [theta[3], 0.0]])
        solver.set_diff_Phi_theta(np.zeros((n, n)), 0)
        solver.set_diff_Phi_theta(np.zeros((n, n)), 1)
        solver.set_diff_Phi_theta([[1.0, 0.0], [0.0, 0.0]], 2)
        solver.set_diff_Phi_theta([[0.0, 0.0], [1.0, 0.0]], 3)

        solver.set_Psi([[theta[0]], [theta[1]]])
        solver.set_diff_Psi_theta([[1.0], [0.0]], 0)
        solver.set_diff_Psi_theta([[0.0], [1.0]], 1)
        solver.set_diff_Psi_theta(np.zeros((n, r)), 2)
        solver.set_diff_Psi_theta(np.zeros((n, r)), 3)

        solver.set_Gamma(np.eye(p))
        solver.set_diff_Gamma_theta(np.zeros((p, p)), 0)
        solver.set_diff_Gamma_theta(np.zeros((p, p)), 1)
        solver.set_diff_Gamma_theta(np.zeros((p, p)), 2)
        solver.set_diff_Gamma_theta(np.zeros((p, p)), 3)

        solver.set_H([[1.0, 0.0]])
        solver.set_diff_H_theta(np.zeros((m, n)), 0)
        solver.set_diff_H_theta(np.zeros((m, n)), 1)
        solver.set_diff_H_theta(np.zeros((m, n)), 2)
        solver.set_diff_H_theta(np.zeros((m, n)), 3)

        solver.set_Q(np.eye(p))
        solver.set_diff_Q_theta(np.zeros((p, p)), 0)
        solver.set_diff_Q_theta(np.zeros((p, p)), 1)
        solver.set_diff_Q_theta(np.zeros((p, p)), 2)
        solver.set_diff_Q_theta(np.zeros((p, p)), 3)

        solver.set_R(0.02 * np.eye(m))
        solver.set_diff_R_theta(np.zeros((m, m)), 0)
        solver.set_diff_R_theta(np.zeros((m, m)), 1)
        solver.set_diff_R_theta(np.zeros((m, m)), 2)
        solver.set_diff_R_theta(np.zeros((m, m)), 3)

        solver.set_x0(np.zeros((n, 1)))
        solver.set_diff_x0_theta(np.zeros((n, 1)), 0)
        solver.set_diff_x0_theta(np.zeros((n, 1)), 1)
        solver.set_diff_x0_theta(np.zeros((n, 1)), 2)
        solver.set_diff_x0_theta(np.zeros((n, 1)), 3)

        solver.set_P0(0.1 * np.eye(n))
        solver.set_diff_P0_theta(np.zeros((n, n)), 0)
        solver.set_diff_P0_theta(np.zeros((n, n)), 1)
        solver.set_diff_P0_theta(np.zeros((n, n)), 2)
        solver.set_diff_P0_theta(np.zeros((n, n)), 3)

        for i in xrange(N):
            solver.set_u([[alpha[i]]], i)

        return solver

    def calc_imf(alpha):
        solver = build_solver(alpha)
        return solver.get_inf_matrix()

    def calc_grad_imf(alpha):
        solver = build_solver(alpha)
        solver.get_inf_matrix()
        return solver.get_diff_inf_matrix(0, 1)

    solver_M  = lambda alpha: calc_imf(alpha)
    solver_dM = lambda alpha: calc_grad_imf(alpha)

    x0 = -1.0; x1 = 1.0
    A = (x1 - x0) * np.random.random((N, q)) + x0      # starting with random plan
    ##A = [[-5, 5], [-5, 5]]
    xi = Plan(N, N, q)
    xi.A[:, :] = A
    xi.set_bounds(((x0, x1), ) * N)
    xi.set_inf_matrix_solver(solver_M)

    build_plan_dirgrad(xi, solver_dM)
    print 'Checking solution for being optimal (less is better): [%.2lf]' % np.abs(xi.mu() - s)

    ##build_plan_dirscan(xi)


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    main()
