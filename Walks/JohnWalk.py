from Walks.BarrierRandomWalk import *

class JohnWalk(BarrierRandomWalk):
    def __init__(self, A, b, x, r):
        super().__init__(A, b, x, r)
        constant = (r ** 2)/(self.d)
        self.term_density = (-0.5/constant)
        self.term_sample = np.sqrt(constant)
    
    def generate_weight(self, x):
        w = self.gradient_descent(x, 0.1)
        return w
    
        '''
        def optimizing_function(input_w):
            term1 = input_w.sum()
            input_W_aj = np.diag(np.power(input_w, a_j))
        
            term2_matrix = np.matmul(np.matmul(S_inverse, input_W_aj), S_inverse)
            term2_matrix = np.matmul(np.matmul(A.T, term2_matrix), A)
            term2 = np.log(np.linalg.det(term2_matrix)) * (-1/a_j)
        
            term3 = (-1/B_j) * np.log(input_w).sum()
            return term1 + term2 + term3
    
        val = minimize(optimizing_function, w_i).x
        print(optimizing_function(val))
        '''
    
    def gradient_descent(self, x, adj, sim = 100):
        alpha = 1 - 1/(np.log2(2 * self.n / self.d))
        beta = self.d / (2 * self.n)
     
        w_i = np.ones(self.n)
        S_inverse = np.diag(1/(self.generate_slack(x)))

        A_x = np.matmul(S_inverse, self.A)
  
        #implement backtracking line search -> decrease step size by half 
        
        for i in range(sim):
            W = np.diag(w_i ** alpha)
            term1 = np.ones(self.n)
            
            term2a = (w_i ** (alpha - 1))
            term2b = np.linalg.inv(np.matmul(np.matmul(A_x.T, W), A_x))
            term2b = np.diag(np.matmul(np.matmul(A_x, term2b), A_x.T))
            
            term2 = np.multiply(term2a, term2b)
            
            term3 = beta * 1/w_i
            
            gradient = term1 - term2 - term3
            
            if np.linalg.norm(gradient) < 0.01:
                break
            
            w_i = w_i - adj * gradient
        return w_i