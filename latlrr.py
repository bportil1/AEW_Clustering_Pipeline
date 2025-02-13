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
        self.mu = 10e-4 # reg term
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
            #print("J: ", self.J)
            #print("S: ", self.S)
            #print("Z: ", self.Z)
            #print("L: ", self.L)
            #print("E: ", self.E)
            #print("Y1: ", self.y1)
            #print("Y2: ", self.y2)
            #print("Y3: ", self.y3)
            #print("MU: ", self.mu)

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
        #singular_vals = np.linalg.svd(matrix, compute_uv = False)
        #return np.max(singular_vals)
        return np.max(np.abs(matrix))

    def orth_projection(self, matrix, indices):
        proj_x = np.zeros_like(matrix)
        for i, j in indices:
            proj_x[i, j] = matrix[i, j]

    '''
    def svt_optimization(self, matrix, max_iter, delta=1.2, l=1, tol=10e-4):
        Y = np.zeros_like(matrix)
        r = 0
        opt_X = np.zeros_like(matrix)
        tau = np.average(np.linalg.svd(matrix, compute_uv=False))
        
        for k in range(1, max_iter+1):
            s = r + 1
            if np.all(Y==0):
                Y = Y + delta * (matrix)
                continue
            while True:
                U, S, Vt = np.linalg.svd(Y, full_matrices=False)
                delta = (len(U[0]) * len(V[0])) / len(matrix[0])

                if s - 1 < len(S) and S[s - l - 1]  <= tau:
                    break
               
                s += l
                    
            r = max(j for j in range(len(S)) if S[j] > tau)

            curr_X = sum((S[j] - tau) * np.outer(U[:, j], Vt[j, :]) for j in range(r))

            if np.linalg.norm(curr_X - matrix, 'fro') /np.linalg.norm(matrix[:r], 'fro') <= tol:
                opt_X = curr_X
                break

            Y = Y + delta * (matrix - curr_X)
            
        return opt_X 
    '''

    def svt_optimization(self, matrix):
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        print("U shape: ", U)
        #print("S shape: ", S)
        print("Vt shape: ", Vt)
        n1 = len(U[0]) 
        n2 = len(Vt[0])
        m = len(matrix[0])
        delta = ((n1 * n2) / m) * 1.2
        #tau = delta * np.linalg.norm(((m*matrix)/(n1*n2)), 'fro' ) 
        tau = np.average(S)
        #print("Current Tau: ", tau)
        S_thresholded = np.maximum(S - tau, 0)
        #print("S_thresholded: ", S_thresholded)
        m, n = matrix.shape
        S_full = np.zeros((m,n))
        np.fill_diagonal(S_full, S_thresholded[:min(m, n)])
        #print("S_Full: ", S_full.shape)
        #print("S_Full: ", S_full)
        return U @ S_full @ Vt

    ######## check multiplication here #########
    def soft_thresholding(self, matrix, tau):
        return np.sign(matrix) * np.maximum(np.abs(matrix) - tau, 0)

    def pg_optimization(self, matrix, max_iter, tol=10e-4):
        #tau = np.average(np.linalg.svd(matrix, compute_uv=False))
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
        #print("ORIG J: ", self.J)
        #A = self.J - (self.Z + (self.y2 / self.mu))
        #print("A: ", A)
        if (self.J.all() != 0):
            A = (self.Z + (self.y2 / self.mu))
            self.J = self.svt_optimization(A)
            #print("THRESHOLDED J: ", self.J)
        #else: 
        #    self.J = A

    def update_S(self):
        print("In update S")
        #A = self.S - (self.L + (self.y3 / self.mu))
        if (self.S.all() != 0):
            A = (self.L + (self.y3 / self.mu))
            self.S = self.svt_optimization(A)
        #else: 
        #    self.S = A

    def update_Z(self):
        #term_1 = np.linalg.inv(self.identity_row + np.dot(self.matrix.T, self.matrix))
        term_1 = self.identity_row + np.matmul(self.matrix.T, self.matrix)

        term_2 = np.matmul(self.matrix.T, (self.matrix - np.matmul(self.L, self.matrix) - self.E))
        term_3 = (np.matmul(self.matrix.T, self.y1) - self.y2) / self.mu
        print("In update_Z")
        #print("term1 shape: ", term_1.shape)
        #print("term2 shape: ", term_2.shape)
        #print("term3 shape: ", term_3.shape)
        #self.Z = np.dot(term_1, (term_2 + self.J + term_3))
        #self.Z = (term_2 + self.J + term_3) / term_1
        self.Z = np.linalg.solve(term_1, (term_2 + self.J + term_3))

        #print("New Z: ", self.Z)

    def update_L(self):
        print("In update L")
        #reg_matrix = 1e-6 * np.eye(self.matrix.shape[0])
        term_1 = np.matmul((self.matrix - np.matmul(self.matrix, self.Z) - self.E), self.matrix.T)
        term_2 = (np.matmul(self.y1, self.matrix.T) - self.y3) / self.mu
        term_3 = np.linalg.pinv(self.identity_col + np.matmul(self.matrix, self.matrix.T))
        
        #term_3 = self.identity_col + np.matmul(self.matrix, self.matrix.T) 
        #term_3 = np.linalg.solve(term_3, np.eye(term_3.shape[0]))
        #self.L = np.matmul((term_1 + self.S + term_2), term_3)
        
        self.L = np.matmul((term_1 + self.S + term_2), term_3)

        #print("New L: ", self.L)

    def update_E(self):
        print("In update E")

        print("J: ", self.J)
        print("S: ", self.S)
        #print("Z: ", self.Z)
        #print("L: ", self.L)
        #print("E: ", self.E)
        #print("Y1: ", self.y1)
        #print("Y2: ", self.y2)
        #print("Y3: ", self.y3)
        #print("MU: ", self.mu)

        #print("first matmul: ", np.matmul(self.matrix, self.Z) )
        #print("second matmul: ", np.matmul(self.L, self.matrix)) 

        print("E computation: ", (self.E - (self.matrix - np.matmul(self.matrix, self.Z) - np.matmul(self.L, self.matrix))))

        A = self.E - ((self.matrix - np.matmul(self.matrix, self.Z) - np.matmul(self.L, self.matrix) + self.y1) / self.mu)
        #A = self.E - ((self.matrix - np.matmul(self.matrix, self.Z) - self.L + self.y1) / self.mu)


        print("A: ", A)

        if (A.all() != 0):
            #self.E = self.pg_optimization(A, 1000)
            tau = np.average(A)
            self.E = self.soft_thresholding(A, tau)

        else:
            self.E = A
        #sigma_max = np.linalg.svd(term_1, compute_uv=False)[0]
        #tau = sigma_max / self.rho
        #self.E = (np.sign(term_1) * np.maximum(np.abs(term_1) - tau, 0))
        print("New E: ", self.E)

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

        print("Current Epsilon: ", self.epsilon)
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


