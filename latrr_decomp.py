import numpy as np

class latLRR():
    def __init__(self, matrix):
        self.matrix = matrix
        self.Z = np.zeros((len(matrix[0]), len(matrix[0])))
        self.J = np.zeros((len(matrix[0]), len(matrix[0])))
        self.L = np.zeros((len(matrix), len(matrix)))
        self.S = np.zeros((len(matrix), len(matrix)))
        self.E = np.zeros((len(matrix), len(matrix)))
        self.identity = np.eye(len(self.matrix[0]))
        self.y1 = 0
        self.y2 = 0
        self.y3 = 0
        self.mew = 10e-6 # reg term
        self.max_u = 10**6
        self.rho = 1.1  # frob norm term  
        self.epsilon = 10e-6
    
    def inexact_alm():
        converged = False
        while not converged:
            self.update_J() 
            self.update_S() 
            self.update_Z()
            self.update_L()
            self.update_E()
            self.update_y1()
            self.update_y2()
            self.update_y3()
            if self.convergence_check():
                converged = True

    def schatten_norm_1(self, matrix):
        singular_vals = np.linalg.svd(matrix, compute_ux = False)
        return np.sum(singular_vals)

    def schatten_norm_inf(self, matrix):
        singular_vals = np.linalg.svd(matrix, compute_uv = False)
        return np.max(singular_vals)

    def sv_thresholding(self, matrix, tau):
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        S_thresholded = np.maximum(S - tau, 0)
        return U @ np.diag(S_thresholded) @ Vt

    def update_J(self):
        tau = self.mew / self.rho
        A = self.Z + (self.y2 / self.mew)
        self.J = self.sv_thresholding(A, tau)

    def update_S(self):
        tau = self.mew / self.rho
        A = self.L + (self.y3 / self.mew)
        self.S = self.sv_thresholding(A, tau)
        
    def update_Z(self):
        term_1 = np.linalg.inv(self.identity + (self.matrix.T * self.matrix))
        term_2 = self.matrix.T * (self.matrix - (self.L * self.matrix) - self.E)
        term_3 = ((self.matrix.T * self.y1) - self.y2) / self.mew
        self.Z = (term_1 * term_2) + self.J + term_3

    def update_L(self):
        term_1 = (self.matrix - (self.matrix * self.Z) - self.E) * self.matrix.T
        term_2 = ((self.y1*self.matrix.T) - self.y3) / self.mew
        term_3 = np.linalg.inv(self.identity + (self.matrix * self.matrix.T))
        self.L = (term_1 + self.S + term_2) * term_3

    def update_E(self):
        tau = self.mew / self.rho
        term_1 = (self.matrix - self.matrix @ self.Z - self.L + self.y1) / self.mew
        self.E = np.sign(matrix) * np.maximum(np.abs(matrix) - tau, 0)

    def update_y1(self):
        self.y1 += self.mew * (self.matrix - (self.matrix * self.Z) - (self.L * self.matrix) - self.E )

    def update_y2(self):
        self.y2 += self.mew * (self.Z - self.J)

    def update_y3(self):
        self.y3 += self.mew * (self.L - self.S)

    def update_mew(self):
        self.mew = np.min((self.rho*self.mew), self.max_u)

    def convergence_check(self):
        test1_term = self.matrix - (self.matrix * self.Z) - (self.L * self.matrix) - self.E
        test1 = self.schatten_norm_inf(test1_term) < self.epsilon

        test2_term = self.Z - self.J
        test2 = self.schatten_norm_inf(test2_term) < self.epsilon

        test3_term = self.L - self.S
        test3 = self.schatten_norm_ing(test3_term) < self.epsilon

        if test1 and test2 and test3:
            return Tru
        else:
            return False


