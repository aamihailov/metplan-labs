# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg as la


# Создать блочную матрицу - столбец [A, dA[0]..dA[N]]
def row_stack_it(A, dA):
    return np.row_stack([A, np.row_stack(dA)])


# Создать блочную матрицу - строку [A, dA[0]..dA[N]]
def col_stack_it(A, dA):
    return np.column_stack([A, np.column_stack(dA)])


# Создать блочную матрицу - строку [O..O, I, O..O], I на i+1 позиции. Размер блока n*n, число блоков s+1
def build_c(n, s, i):
    return np.column_stack([np.eye(n) if i == j else np.zeros((n, n)) for j in xrange(s + 1)])


class IMFSolver(object):
    def __init__(self, n, r, p, m, s, N):
        self.n = n                  # Размерность вектора состояний x
        self.r = r                  # Размерность вектора управления u
        self.p = p                  # Размерность вектора возмущений w
        self.m = m                  # Размерность вектора измерений y и вектора ошибки измерения v

        self.N = N                  # Число наблюдений: t=0..N
        self.s = s                  # Число параметров theta
        self._init_structures()

    def _init_structures(self):
        data = dict()
        data['n'] = self.n
        data['r'] = self.r
        data['p'] = self.p
        data['m'] = self.m

        data['N'] = self.N
        data['s'] = self.s

        # Матрица состояния
        data['Phi'] = np.ndarray((self.n, self.n))
        for i in xrange(self.s):
            data['diff(Phi, theta[%d])' % i] = np.ndarray((self.n, self.n))

        # Матрица управления
        data['Psi'] = np.ndarray((self.n, self.r))
        for i in xrange(self.s):
            data['diff(Psi, theta[%d])' % i] = np.ndarray((self.n, self.r))

        # Матрица возмущения
        data['Gamma'] = np.ndarray((self.n, self.p))
        for i in xrange(self.s):
            data['diff(Gamma, theta[%d])' % i] = np.ndarray((self.n, self.p))

        # Матрица измерений
        data['H'] = np.ndarray((self.m, self.n))
        for i in xrange(self.s):
            data['diff(H, theta[%d])' % i] = np.ndarray((self.m, self.n))

        data['Q'] = np.ndarray((self.p, self.p))
        for i in xrange(self.s):
            data['diff(Q, theta[%d])' % i] = np.ndarray((self.p, self.p))

        data['R'] = np.ndarray((self.m, self.m))
        for i in xrange(self.s):
            data['diff(R, theta[%d])' % i] = np.ndarray((self.m, self.m))

        data['x(0)'] = np.ndarray((self.n, 1))
        for i in xrange(self.s):
            data['diff(x(0), theta[%d])' % i] = np.ndarray((self.n, 1))

        data['P(0)'] = np.ndarray((self.n, self.n))
        for i in xrange(self.s):
            data['diff(P(0), theta[%d])' % i] = np.ndarray((self.n, self.n))

        data['M'] = np.ndarray((self.s, self.s))

        for i in xrange(self.N):
            data['u(%d)' % i] = np.ndarray((self.r, 1))

        self.data = data

    def set_Phi(self, A):
        self.data['Phi'][:, :] = A

    def set_diff_Phi_theta(self, A, i):
        self.data['diff(Phi, theta[%d])' % i][:, :] = A

    def set_Psi(self, A):
        self.data['Psi'][:, :] = A

    def set_diff_Psi_theta(self, A, i):
        self.data['diff(Psi, theta[%d])' % i][:, :] = A

    def set_Gamma(self, A):
        self.data['Gamma'][:, :] = A

    def set_diff_Gamma_theta(self, A, i):
        self.data['diff(Gamma, theta[%d])' % i][:, :] = A

    def set_H(self, A):
        self.data['H'][:, :] = A

    def set_diff_H_theta(self, A, i):
        self.data['diff(H, theta[%d])' % i][:, :] = A

    def set_Q(self, A):
        self.data['Q'][:, :] = A

    def set_diff_Q_theta(self, A, i):
        self.data['diff(Q, theta[%d])' % i][:, :] = A

    def set_R(self, A):
        self.data['R'][:, :] = A

    def set_diff_R_theta(self, A, i):
        self.data['diff(R, theta[%d])' % i][:, :] = A

    def set_x0(self, A):
        self.data['x(0)'][:, :] = A

    def set_diff_x0_theta(self, A, i):
        self.data['diff(x(0), theta[%d])' % i][:, :] = A

    def set_P0(self, A):
        self.data['P(0)'][:, :] = A

    def set_diff_P0_theta(self, A, i):
        self.data['diff(P(0), theta[%d])' % i][:, :] = A

    def set_u(self, A, t):
        self.data['u(%d)' % t][:, :] = A

    def get_inf_matrix(self):
        d = self.data
        # Шаг 1. Сформировать матрицу Psi_A в соответствии с равенством (2.77)
        d['Psi_A'] = row_stack_it(d['Psi'], [d['diff(Psi, theta[%d])' % i] for i in xrange(self.s)])

        # Шаг 2.
        d['M(Theta)'] = np.ndarray((self.s, self.s))
        d['M(Theta)'][:, :] = 0.0
        d['P(0|0)'] = d['P(0)']
        for i in xrange(self.s):
            d['diff(P(0|0), theta[%d])' % i] = d['diff(P(0), theta[%d])' % i]
        t = 0

        while t < self.N:
            self.step3(d, t)
            self.step4(d, t)
            self.step5(d, t)
            self.step6(d, t)
            self.step7(d, t)
            self.step8(d, t)
            self.step9(d, t)
            # Шаг 10. Положить M(Theta) = M(Theta) + delta M(Theta)
            d['M(Theta)'] += d['delta M(Theta)']

            # Шаг 11. Увеличить t на единицу. Если t <= N-1, перейти на шаг 3. В противном случае закончить процесс
            t += 1
        return d['M(Theta)']

    def get_diff_inf_matrix(self, j, tau):
        d = self.data

        d['Phi_t_A'] = row_stack_it(d['Phi'], [d['diff(Phi, theta[%d])' % i] for i in xrange(self.s)])

        d['diff(M(U, Theta), u(%d, %d))' % (j, tau)] = np.ndarray((self.s, self.s))
        d['diff(M(U, Theta), u(%d, %d))' % (j, tau)][:, :] = 0.0

        t = 0
        while t < self.N:
            self.diff_step3(d, t, j, tau)
            self.diff_step6(d, t, j, tau)

            d['diff(M(U, Theta), u(%d, %d))' % (j, tau)] += d['delta diff(M(U, Theta), u(%d, %d))' % (j, tau)]
            # Шаг 8. Увеличить t на единицу. Если t <= N-1, перейти на шаг 3. В противном случае закончить процесс
            t += 1

        return d['diff(M(U, Theta), u(%d, %d))' % (j, tau)]

    def step3(self, d, t):
        # Шаг 3. Вычислить Sigma_A(t + 1|t) по формуле (2.82), если t = 0, иначе по формуле (2.84)
        if t == 0:
            d['Sigma_A(%d|%d)' % (t + 1, t)] = 0
        else:
            Phi = d['Phi_A(%d|%d)' % (t + 1, t)]
            Sigma = d['Sigma_A(%d|%d)' % (t, t - 1)]
            K = d['K_A(%d)' % t]
            B = d['B(%d)' % t]
            d['Sigma_A(%d|%d)' % (t + 1, t)] = np.dot(np.dot(Phi, Sigma), Phi.transpose()) + np.dot(np.dot(K, B), K.transpose())

    def step4(self, d, t):
        # Шаг 4. Определить x(t + 1|t) при помощи выражения (2.81)
        if t == 0:
            P1 = row_stack_it(d['Phi'], [d['diff(Phi, theta[%d])' % i] for i in xrange(self.s)])
            x0 = d['x(0)']
            z0 = np.ndarray((self.n, 1))
            z0[:, :] = 0
            P2 = row_stack_it(z0, [np.dot(d['Phi'], d['diff(x(0), theta[%d])' % i]) for i in xrange(self.s)])
            d['x_A(%d|%d)' % (t + 1, t)] = np.dot(P1, x0) + P2 + np.dot(d['Psi_A'], d['u(0)'])
        else:
            d['x_A(%d|%d)' % (t + 1, t)] = \
                np.dot(d['Phi_A(%d|%d)' % (t + 1, t)], d['x_A(%d|%d)' % (t, t - 1)]) + \
                np.dot(d['Psi_A'], d['u(%d)' % t])

    def step5(self, d, t):
        # Шаг 5.Найти P(t + 1|t), B(t + 1), K(t + 1), P(t + 1|t + 1), Kt(t + 1),
        # используя соотношения (2.9), (2.11), (2.12), (2.14) и (2.73)
        self.step5_P10(d, t)
        self.step5_B(d, t)
        self.step5_K(d, t)
        self.step5_P11(d, t)
        self.step5_Kt(d, t)

    def step5_P10(self, d, t):
        Phi = d['Phi']
        P = d['P(%d|%d)' % (t, t)]
        Gamma = d['Gamma']
        Q = d['Q']
        d['P(%d|%d)' % (t + 1, t)] = np.dot(np.dot(Phi, P), Phi.transpose()) + np.dot(np.dot(Gamma, Q), Gamma.transpose())

    def step5_B(self, d, t):
        H = d['H']
        P = d['P(%d|%d)' % (t + 1, t)]
        R = d['R']
        d['B(%d)' % (t + 1)] = np.dot(np.dot(H, P), H.transpose()) + R

    def step5_K(self, d, t):
        P = d['P(%d|%d)' % (t + 1, t)]
        H = d['H']
        B = d['B(%d)' % (t + 1)]
        d['K(%d)' % (t + 1)] = np.dot(np.dot(P, H.transpose()), la.inv(B))

    def step5_P11(self, d, t):
        I = np.eye(self.n)
        K = d['K(%d)' % (t + 1)]
        H = d['H']
        P = d['P(%d|%d)' % (t + 1, t)]
        d['P(%d|%d)' % (t + 1, t + 1)] = np.dot((I - np.dot(K, H)), P)

    def step5_Kt(self, d, t):
        Phi = d['Phi']
        K = d['K(%d)' % (t + 1)]
        d['Kt(%d)' % (t + 1)] = np.dot(Phi, K)

    def step6(self, d, t):
        # Шаг 6. Сформировать матрицу Phi_A(t+2|t+1) в соответствии с (2.76)
        First_col = row_stack_it(d['Phi'], [d['diff(Phi, theta[%d])' % i] -
                                            np.dot(d['Kt(%d)' % (t + 1)], d['diff(H, theta[%d])' % i]) for i in xrange(self.s)])
        Phi = d['Phi']
        Kt = d['Kt(%d)' % (t + 1)]
        H = d['H']
        Remain_cols = [row_stack_it(np.zeros((self.n, self.n)), [Phi - np.dot(Kt, H) if i == j else np.zeros((self.n, self.n))
                                                                 for j in xrange(self.s)]) for i in xrange(self.s)]
        d['Phi_A(%d|%d)' % (t + 2, t + 1)] = col_stack_it(First_col, Remain_cols)

    def step7(self, d, t):
        # Шаг 7. Вычислить производные для P(t+1|t), B(t+1), K(t+1), P(t+1|t+1), Kt(t+1) по формулам (2.85) - (2.89)
        self.step7_P10(d, t)
        self.step7_B(d, t)
        self.step7_K(d, t)
        self.step7_P11(d, t)
        self.step7_Kt(d, t)

    def step7_P10(self, d, t):
        for i in xrange(self.s):
            dPhi = d['diff(Phi, theta[%d])' % i]
            P = d['P(%d|%d)' % (t, t)]
            Phi = d['Phi']
            dP = d['diff(P(%d|%d), theta[%d])' % (t, t, i)]
            dGamma = d['diff(Gamma, theta[%d])' % i]
            Q = d['Q']
            Gamma = d['Gamma']
            dQ = d['diff(Q, theta[%d])' % i]
            d['diff(P(%d|%d), theta[%d])' % (t + 1, t, i)] = \
                np.dot(np.dot(dPhi, P), Phi.transpose()) + \
                np.dot(np.dot(Phi, dP), Phi.transpose()) + \
                np.dot(np.dot(Phi, P), dPhi.transpose()) + \
                np.dot(np.dot(dGamma, Q), Gamma.transpose()) + \
                np.dot(np.dot(Gamma, dQ), Gamma.transpose()) + \
                np.dot(np.dot(Gamma, Q), dGamma.transpose())

    def step7_B(self, d, t):
        for i in xrange(self.s):
            dH = d['diff(H, theta[%d])' % i]
            P = d['P(%d|%d)' % (t + 1, t)]
            H = d['H']
            dP = d['diff(P(%d|%d), theta[%d])' % (t + 1, t, i)]
            dR = d['diff(R, theta[%d])' % i]
            d['diff(B(%d), theta[%d])' % (t + 1, i)] = \
                np.dot(np.dot(dH, P), H.transpose()) + \
                np.dot(np.dot(H, dP), H.transpose()) + \
                np.dot(np.dot(H, P), dH.transpose()) + dR

    def step7_K(self, d, t):
        for i in xrange(self.s):
            dP = d['diff(P(%d|%d), theta[%d])' % (t + 1, t, i)]
            H = d['H']
            P = d['P(%d|%d)' % (t + 1, t)]
            dH = d['diff(H, theta[%d])' % i]
            B = d['B(%d)' % (t + 1)]
            dB = d['diff(B(%d), theta[%d])' % (t + 1, i)]
            d['diff(K(%d), theta[%d])' % (t + 1, i)] = np.dot(
                np.dot(dP, H.transpose()) +
                np.dot(P, dH.transpose()) -
                np.dot(np.dot(np.dot(P, H.transpose()), la.inv(B)), dB), la.inv(B))

    def step7_P11(self, d, t):
        for i in xrange(self.s):
            I = np.eye(self.n)
            K = d['K(%d)' % (t + 1)]
            H = d['H']
            dP = d['diff(P(%d|%d), theta[%d])' % (t + 1, t, i)]
            dK = d['diff(K(%d), theta[%d])' % (t + 1, i)]
            dH = d['diff(H, theta[%d])' % i]
            P = d['P(%d|%d)' % (t + 1, t)]
            d['diff(P(%d|%d), theta[%d])' % (t + 1, t + 1, i)] = \
                np.dot((I - np.dot(K, H)), dP) - \
                np.dot(np.dot(dK, H) + np.dot(K, dH), P)

    def step7_Kt(self, d, t):
        for i in xrange(self.s):
            dPhi = d['diff(Phi, theta[%d])' % i]
            K = d['K(%d)' % (t + 1)]
            Phi = d['Phi']
            dK = d['diff(K(%d), theta[%d])' % (t + 1, i)]
            d['diff(Kt(%d), theta[%d])' % (t + 1, i)] = np.dot(dPhi, K) + np.dot(Phi, dK)

    def step8(self, d, t):
        # Шаг 8. Сформировать матрицу K_A(t+1) в соответствии с (2.78)
        d['K_A(%d)' % (t + 1)] = row_stack_it(d['Kt(%d)' % (t + 1)],
                                              [d['diff(Kt(%d), theta[%d])' % (t + 1, i)] for i in xrange(self.s)])

    def step9(self, d, t):
        # Шаг 9. Используя выражение (2.72), получить приращение deltaM(Theta), отвечающее текущему значению t.
        deltaM = np.ndarray((self.s, self.s))
        C0 = build_c(self.n, self.s, 0)
        Sigma_A = d['Sigma_A(%d|%d)' % (t + 1, t)]
        x_A = d['x_A(%d|%d)' % (t + 1, t)]
        B = d['B(%d)' % (t + 1)]
        H = d['H']
        for i in xrange(self.s):
            Ci = build_c(self.n, self.s, i + 1)
            dHi = d['diff(H, theta[%d])' % i]
            dBi = d['diff(B(%d), theta[%d])' % (t + 1, i)]
            for j in xrange(self.s):
                Cj = build_c(self.n, self.s, j + 1)
                dHj = d['diff(H, theta[%d])' % j]
                dBj = d['diff(B(%d), theta[%d])' % (t + 1, j)]

                deltaM[i, j] = \
                    np.trace(np.dot(np.dot(np.dot(np.dot(np.dot(C0, (Sigma_A + np.dot(x_A, x_A.transpose()))), C0.transpose()), dHj.transpose()), la.inv(B)), dHi)) + \
                    np.trace(np.dot(np.dot(np.dot(np.dot(np.dot(C0, (Sigma_A + np.dot(x_A, x_A.transpose()))), Cj.transpose()), H.transpose()), la.inv(B)), dHi)) + \
                    np.trace(np.dot(np.dot(np.dot(np.dot(np.dot(Ci, (Sigma_A + np.dot(x_A, x_A.transpose()))), C0.transpose()), dHj.transpose()), la.inv(B)), H)) + \
                    np.trace(np.dot(np.dot(np.dot(np.dot(np.dot(Ci, (Sigma_A + np.dot(x_A, x_A.transpose()))), Cj.transpose()), H.transpose()), la.inv(B)), H)) + \
                    0.5 * np.trace(np.dot(np.dot(dBi, la.inv(B)), np.dot(dBj, la.inv(B))))
        d['delta M(Theta)'] = deltaM

    def diff_step3(self, d, t, j, tau):
        # Шаг 3. Вычислить производную x_A
        dx = 'diff(x_A(%d|%d), u(%d, %d))' % (t + 1, t, j, tau)
        dxprev = 'diff(x_A(%d|%d), u(%d, %d))' % (t, t - 1, j, tau)
        if t == 0:
            if tau == 0:
                d[dx] = np.ndarray((self.n * (self.s + 1), 1))
            else:
                d[dx] = np.array([d['Psi_A'][:, j]]).transpose()
        else:
            Phi = d['Phi_A(%d|%d)' % (t + 1, t)]
            dx_A = d['diff(x_A(%d|%d), u(%d, %d))' % (t, t - 1, j, tau)]
            Psi = d['Psi_A']

            du = np.zeros((self.r, 1))
            if t == tau:
                du[j, 0] = 1.0

            d[dx] = np.dot(Phi, dx_A) + np.dot(Psi, du)

    def diff_step6(self, d, t, j, tau):
        deltaM = np.ndarray((self.s, self.s))
        C0 = build_c(self.n, self.s, 0)
        x_A = d['x_A(%d|%d)' % (t + 1, t)]
        dx_A = d['diff(x_A(%d|%d), u(%d, %d))' % (t + 1, t, j, tau)]
        B = d['B(%d)' % (t + 1)]
        H = d['H']
        for alpha in xrange(self.s):
            Ci = build_c(self.n, self.s, alpha + 1)
            dHi = d['diff(H, theta[%d])' % alpha]
            dBi = d['diff(B(%d), theta[%d])' % (t + 1, alpha)]
            for beta in xrange(self.s):
                Cj = build_c(self.n, self.s, beta + 1)
                dHj = d['diff(H, theta[%d])' % j]
                dBj = d['diff(B(%d), theta[%d])' % (t + 1, beta)]

                x_dx_p_dx_x = np.dot(x_A, dx_A.transpose()) + np.dot(dx_A, x_A.transpose())
                deltaM[alpha, beta] = \
                    np.trace(np.dot(np.dot(np.dot(np.dot(np.dot(C0, x_dx_p_dx_x), C0.transpose()), dHj.transpose()), la.inv(B)), dHi)) + \
                    np.trace(np.dot(np.dot(np.dot(np.dot(np.dot(C0, x_dx_p_dx_x), Cj.transpose()), H.transpose()), la.inv(B)), dHi)) + \
                    np.trace(np.dot(np.dot(np.dot(np.dot(np.dot(Ci, x_dx_p_dx_x), C0.transpose()), dHj.transpose()), la.inv(B)), H)) + \
                    np.trace(np.dot(np.dot(np.dot(np.dot(np.dot(Ci, x_dx_p_dx_x), Cj.transpose()), H.transpose()), la.inv(B)), H))
        d['delta diff(M(U, Theta), u(%d, %d))' % (j, tau)] = deltaM


def main():
    N = 20

    solver = IMFSolver(n=2, r=1, p=2, m=1, s=2, N=N)

    theta = [0.56, 0.48]

    solver.set_Phi([[1.0, 1.0], [-0.5, 0.0]])
    solver.set_diff_Phi_theta([[0.0, 0.0], [0.0, 0.0]], 0)
    solver.set_diff_Phi_theta([[0.0, 0.0], [0.0, 0.0]], 1)

    solver.set_Psi([[theta[0]], [theta[1]]])
    solver.set_diff_Psi_theta([[1.0], [0.0]], 0)
    solver.set_diff_Psi_theta([[0.0], [1.0]], 1)

    solver.set_Gamma([[1.0, 0.0], [0.0, 1.0]])
    solver.set_diff_Gamma_theta([[0.0, 0.0], [0.0, 0.0]], 0)
    solver.set_diff_Gamma_theta([[0.0, 0.0], [0.0, 0.0]], 1)

    solver.set_H([[1.0, 0.0]])
    solver.set_diff_H_theta([[0.0, 0.0]], 0)
    solver.set_diff_H_theta([[0.0, 0.0]], 1)

    solver.set_Q([[0.07, 0.0], [0.0, 0.07]])
    solver.set_diff_Q_theta([[0.0, 0.0], [0.0, 0.0]], 0)
    solver.set_diff_Q_theta([[0.0, 0.0], [0.0, 0.0]], 1)

    solver.set_R([[0.02]])
    solver.set_diff_R_theta([[0.0]], 0)
    solver.set_diff_R_theta([[0.0]], 1)

    solver.set_x0([[0.0], [0.0]])
    solver.set_diff_x0_theta([[0.0], [0.0]], 0)
    solver.set_diff_x0_theta([[0.0], [0.0]], 1)

    solver.set_P0([[0.1, 0.0], [0.0, 0.1]])
    solver.set_diff_P0_theta([[0.0, 0.0], [0.0, 0.0]], 0)
    solver.set_diff_P0_theta([[0.0, 0.0], [0.0, 0.0]], 1)

    for i in xrange(N):
        solver.set_u([[1.0]], i)

    M = solver.get_inf_matrix()
    print M
    print la.det(M)
    print -np.log(la.det(M))

    dM = solver.get_diff_inf_matrix(0, 10)
    print dM
    print np.trace(np.dot(la.inv(M), dM))

if __name__ == '__main__':
    main()