from BarrierRandomWalk import *

class DikinLSWalk(BarrierRandomWalk):
    def __init__(self, A, b, x, r):
        super().__init__(A, b, x, r)
        constant = (r ** 2)/self.d
        self.term_density = (-0.5/constant)
        self.term_sample = np.sqrt(constant)
    
    def generate_weight(self, x):
        w_i = np.ones(self.n)
        q = 2 * (1 + np.log(self.n))
        
        S_inv = np.diag(1/self.generate_slack(x))
        A_x = np.matmul(S_inv, self.A)
        W = np.diag(w_i ** (1 - (2/q)))
        
        return self.gradient_descent(x, 0.1)
        '''
        def optimizing_function(input_w):
            W = np.diag(input_w ** (1 - (2/q)))
            term1 = np.matmul(A_x.T, np.matmul(W, A_x))
            term1 = np.log(np.linalg.det(term1))
            term2 = (0.5 - (1/q)) * input_w.sum()
            return -1 * (term1 - term2)
        
        matrix_identity = np.identity(self.n)
        array_zeroes = np.repeat(0.0001, self.n)
        array_infinite = np.repeat(np.inf, self.n)
        
        linear_constraint = LinearConstraint(matrix_identity, array_zeroes, array_infinite)
        
        return minimize(optimizing_function, w_i, constraints = linear_constraint).x
        '''
    
    def gradient_descent(self, x, adj, sim = 100):
        q = 2 * (1 + np.log(self.n))
        alpha = 1 - (2/q)
        
        w_i = np.ones(self.n)
        S_inverse = np.diag(1/(self.generate_slack(x)))
        A_x = np.matmul(S_inverse, self.A)
        
        for i in range(sim):
            W = np.diag(w_i ** alpha)
            
            term1a = (alpha * w_i ** (alpha - 1))
            term1b = np.linalg.inv(np.matmul(np.matmul(A_x.T, W), A_x))
            term1b = np.diag(np.matmul(np.matmul(A_x, term1b), A_x.T))
            
            term1 = np.multiply(term1a, term1b)
            term2 = np.ones(self.n) * (0.5 - 1/q)
            
            gradient = term1 - term2
            if np.linalg.norm(gradient) < 0.01:
                break
            
            proposal = w_i + adj * gradient

            if (proposal < 0).any():
                break
                
            w_i = proposal
        
        return w_i