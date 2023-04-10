from BarrierRandomWalk import *

class VaidyaWalk(BarrierRandomWalk):
    def __init__(self, A, b, x, r):
        super().__init__(A, b, x, r)
        constant = (r ** 2)/(np.sqrt(self.n * self.d))
        self.term_density = (-0.5/constant)
        self.term_sample = np.sqrt(constant)
    
    def generate_weight(self, x):
        dikin_matrix_inv, slack_inv = self.generate_dikin_hessian(x)
        dikin_matrix_inv = np.linalg.inv(dikin_matrix_inv)
        
        weights = np.matmul(np.matmul(self.A, dikin_matrix_inv), self.A.T)
        weights = np.matmul(np.matmul(slack_inv, weights), slack_inv)

        
        return np.diagonal(weights) + self.d/self.n
        
        #diag( S-1 A  D-1 A^T S-1 )
        
    def generate_dikin_hessian(self, x):
        slack_inv = np.diag(1/self.generate_slack(x))
        matrix = np.matmul(slack_inv, slack_inv)
        matrix = np.matmul(np.matmul(self.A.T, matrix), self.A)
        return matrix, slack_inv