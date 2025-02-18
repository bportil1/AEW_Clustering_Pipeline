import numpy as np
from scipy import sparse

class latLRR():
    def __init__(self, matrix):
        self.matrix = matrix
        self.Z = np.zeros((len(matrix[0]), len(matrix[0])))
        self.J = np.zeros((len(matrix[0]), len(matrix[0])))
        self.L = np.zeros((len(matrix), len(matrix)))
        self.S = np.zeros((len(matrix), len(matrix)))
        self.E = np.zeros_like(self.matrix)
        self.identity_row = np.eye(len(self.matrix[0]))
        self.identity_col = np.eye(len(self.matrix))
        self.y1 = np.zeros_like(self.matrix)
        self.y2 = np.zeros_like(self.Z)
        self.y3 = np.zeros_like(self.L)
        self.mu = 1e-6 # reg term
        self.max_u = 1e6
        self.rho = 1.1  # frob norm term  
        self.epsilon = 10e-6
    
    def inexact_alm(self):
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
            self.update_mu()

            if self.convergence_check():
                converged = True
                print("Shape of Z: ", self.Z.shape)
                print("Shape of L: ", self.L.shape)
            else:
                print("Not Converged Continuing")

    def schatten_norm_1(self, matrix):
        singular_vals = np.linalg.svd(matrix, compute_ux = False)
        return np.sum(singular_vals)

    def schatten_norm_inf(self, matrix):
        return np.max(np.abs(matrix))

    def orth_projection(self, matrix, indices):
        proj_x = np.zeros_like(matrix)
        for i, j in indices:
            proj_x[i, j] = matrix[i, j]
            
    def svt_optimization(self, matrix):
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        n1 = len(U[0]) 
        n2 = len(Vt[0])
        m = len(matrix[0])
        delta = ((n1 * n2) / m) * 1.2
        tau = np.average(S)
        S_thresholded = np.maximum(S - tau, 0)
        m, n = matrix.shape
        S_full = np.zeros((m,n))
        np.fill_diagonal(S_full, S_thresholded[:min(m, n)])
        return U @ S_full @ Vt

    ######## check multiplication here #########
    def soft_thresholding(self, matrix, tau):
        return np.sign(matrix) * np.maximum(np.abs(matrix) - tau, 0)

    def pg_optimization(self, matrix, max_iter, tol=10e-4):
        tau = .9 / self.mu
        curr_E = np.zeros_like(matrix)
        
        for k in range(max_iter):
            last_E = np.copy(curr_E)
            curr_E = self.soft_thresholding(matrix, tau) 
            if np.linalg.norm(matrix - last_E, ord='fro') < tol:
                break
        return last_E
        

    def update_J(self):
        print("In update J")
        if (self.J.all() != 0):
            A = (self.Z + (self.y2 / self.mu))
            self.J = self.svt_optimization(A)
        else: 
            self.J = (self.Z + (self.y2 / self.mu))

    def update_S(self):
        print("In update S")
        if (self.S.all() != 0):
            A = (self.L + (self.y3 / self.mu))
            self.S = self.svt_optimization(A)
        else: 
            self.S = (self.L + (self.y3 / self.mu))

    def update_Z(self):
        term_1 = self.identity_row + np.matmul(self.matrix.T, self.matrix)

        term_2 = np.matmul(self.matrix.T, (self.matrix - np.matmul(self.L, self.matrix) - self.E))
        term_3 = (np.matmul(self.matrix.T, self.y1) - self.y2) / self.mu
        print("In update_Z")
        self.Z = np.linalg.solve(term_1, (term_2 + self.J + term_3))

    def update_L(self):
        print("In update L")
        term_1 = np.matmul((self.matrix - np.matmul(self.matrix, self.Z) - self.E), self.matrix.T)
        term_2 = (np.matmul(self.y1, self.matrix.T) - self.y3) / self.mu
        term_3 = np.linalg.pinv(self.identity_col + np.matmul(self.matrix, self.matrix.T))
                
        self.L = np.matmul((term_1 + self.S + term_2), term_3)

    def update_E(self):
        print("In update E")

        A = self.matrix - np.matmul(self.matrix, self.Z) - np.matmul(self.L, self.matrix) + self.y1 / self.mu

        if (A.all() != 0):
            tau = np.average(A)
            self.E = self.soft_thresholding(A, tau)

        else:
            self.E = A
            
    def update_y1(self):
        self.y1 += self.mu * (self.matrix - np.matmul(self.matrix, self.Z) - np.matmul(self.L, self.matrix) - self.E )

    def update_y2(self):
        self.y2 += self.mu * (self.Z - self.J)

    def update_y3(self):
        self.y3 += self.mu * (self.L - self.S)

    def update_mu(self):
        self.mu = np.minimum((self.mu*self.rho), self.max_u)
        #print("New Mu: ", self.mu)

    def convergence_check(self):
        test1_term = self.matrix - np.matmul(self.matrix, self.Z) - np.matmul(self.L, self.matrix) - self.E
        test1_res = self.schatten_norm_inf(test1_term)
        test1 = test1_res < self.epsilon

        print("Current Mu: ", self.mu)

        print("Test 1 Value: ", test1_res)

        test2_term = self.Z - self.J
        test2_res = self.schatten_norm_inf(test2_term) 
        test2 = test2_res < self.epsilon

        print("Test 2 Value: ", test2_res)

        test3_term = self.L - self.S
        test3_res = self.schatten_norm_inf(test3_term) 
        test3 = test3_res < self.epsilon

        print("Test 3 Value: ", test3_res)

        if test1 and test2 and test3:
            return True
        else:
            return False


