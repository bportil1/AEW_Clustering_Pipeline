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
        self.mu = 10e-6 # reg term
        self.max_u = 10e6
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
            print("Not Converged Continuing")

    def schatten_norm_1(self, matrix):
        #print("term1 shape: ", term_1.shape)
        #print("term2 shape: ", term_2.shape)
        #print("term3 shape: ", term_3.shape)
        singular_vals = np.linalg.svd(matrix, compute_ux = False)
        return np.sum(singular_vals)

    def schatten_norm_inf(self, matrix):
        singular_vals = np.linalg.svd(matrix, compute_uv = False)
        return np.max(singular_vals)

    def svt(self, matrix):
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        tau = S[0] / self.rho
        #print("Current Tau: ", tau)
        #print("S init: ", S)
        #print("S Length: ", len(S))
        S_thresholded = np.maximum(S - tau, 0)
        #print("S threshed: ", S_thresholded)
        #print("S Threshed Size: ", len(S_thresholded))
        return U @ np.diag(S_thresholded) @ Vt

    def orth_projection(self, matrix, indices):
        proj_x = np.zeros_like(matrix)
        for i, j in indices:
            proj_x[i, j] = matrix[i, j]

    def svt_optimization(self, matrix, max_iter, delta=1.2, l=1, tol=10e-4):
        Y = np.zeros_like(matrix)
        r = 0
        opt_X = np.zeros_like(matrix)
        #tau = np.minimum(len(matrix), len(matrix[0])) * 5
        tau = np.average(np.linalg.svd(matrix, compute_uv=False))
        print("Tau: ", tau)
        for k in range(1, max_iter+1):
            s = r + 1
            #while True:
            #print("Y: ", Y)
            #print("Y: ", np.all(Y==0))
            if np.all(Y==0):
                Y = Y + delta * (matrix)
                continue
            while True:
                U, S, Vt = np.linalg.svd(Y, full_matrices=False)
                #print("U: ", U)
                #print("S: ", S)
                #print("Vt: ", Vt)

                if s - 1 < len(S) and S[s - l - 1]  <= tau:
                    break
                s += 1
                    
            r = max(j for j in range(len(S)) if S[j] > tau)
            curr_X = sum((S[j] - tau) * np.outer(U[:, j], Vt[j, :]) for j in range(r))

            print("Current Convergence Condition: ", np.linalg.norm(curr_X - matrix, 'fro') /np.linalg.norm(matrix, 'fro'))

            if np.linalg.norm(curr_X - matrix, 'fro') /np.linalg.norm(matrix, 'fro') <= tol:
                opt_X = curr_X
                break

            Y = Y + delta * (matrix - curr_X)
            
        return opt_X 

    def update_J(self):
        print("In update J")
        #tau = self.mu / self.rho
        #print("Curr Tau: ", tau)
        A = self.J - (self.Z + (self.y2 / self.mu))
        #print("A: ", A)
        #self.J = self.svt(A)
        if (A.all() != 0):
            self.J = self.svt_optimization(A, 1000)
        else: 
            self.J = A
        #print("New J: ", self.J)

    def update_S(self):
        print("In update S")
        #tau = self.mu / self.rho
        A = self.S - (self.L + (self.y3 / self.mu))
        self.S = self.svt(A)
        #print("New S: ", self.S)

    def update_Z(self):
        term_1 = np.linalg.inv(self.identity_row + np.dot(self.matrix.T, self.matrix))
        term_2 = np.dot(self.matrix.T, (self.matrix - np.dot(self.L, self.matrix) - self.E))
        term_3 = (np.dot(self.matrix.T, self.y1) - self.y2) / self.mu
        print("In update_Z")
        #print("term1 shape: ", term_1.shape)
        #print("term2 shape: ", term_2.shape)
        #print("term3 shape: ", term_3.shape)
        self.Z = np.dot(term_1, (term_2 + self.J + term_3))
        #print("New Z: ", self.Z)

    def update_L(self):
        term_1 = np.dot((self.matrix - np.dot(self.matrix, self.Z) - self.E), self.matrix.T)
        term_2 = (np.dot(self.y1, self.matrix.T) - self.y3) / self.mu
        term_3 = np.linalg.inv(self.identity_col + np.dot(self.matrix, self.matrix.T))
        print("In update L")
        self.L = np.dot((term_1 + self.S + term_2), term_3)
        #print("New L: ", self.L)

    def update_E(self):
        print("In update E")
        term_1 = self.E - ((self.matrix - np.dot(self.matrix, self.Z) - np.dot(self.L, self.matrix) + self.y1) / self.mu)
        sigma_max = np.linalg.svd(term_1, compute_uv=False)[0]
        tau = sigma_max / self.rho
        self.E = (np.sign(term_1) * np.maximum(np.abs(term_1) - tau, 0))
        #print("New E: ", self.E)

    def update_y1(self):
        self.y1 += self.mu * (self.matrix - np.dot(self.matrix, self.Z) - np.dot(self.L, self.matrix) - self.E )

    def update_y2(self):
        self.y2 += self.mu * (self.Z - self.J)

    def update_y3(self):
        self.y3 += self.mu * (self.L - self.S)

    def update_mu(self):
        #print("Current Mu: ", self.mu)
        #print("Current Rho: ", self.rho)
        #print("Current Max_u: ", self.max_u)
        self.mu = np.minimum((self.rho*self.mu), self.max_u)
        #print("New Mu: ", self.mu)

    def convergence_check(self):
        test1_term = self.matrix - np.dot(self.matrix, self.Z) - np.dot(self.L, self.matrix) - self.E
        test1_res = self.schatten_norm_inf(test1_term)
        test1 = test1_res < self.epsilon

        print("Current Epsilon: ", self.epsilon)

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


