import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.optimize import minimize, LinearConstraint, linprog
import pandas as pd
import sympy

class BarrierRandomWalk:
    def __init__(self, A, b, x, r):
        
        assert np.linalg.matrix_rank(A) == A.shape[1]
        
        self.A = A
        self.b = b
        self.x = x
        self.r = r
        
        self.n = A.shape[0]
        self.d = A.shape[1]
        
        self.term_sample = 1
        self.term_density = 1
    
    def accept_reject(self, z):
        return (np.matmul(self.A,z) < self.b).all()
    
    def local_norm(self, vector, matrix):
        return np.matmul(np.matmul(vector.T, matrix), vector).item()
    
    def generate_gaussian_rv(self):
        return np.random.randn(self.d)[..., None]
    
    def generate_slack(self, x):
        v = (self.b - np.matmul(self.A,x)).T
        return np.squeeze(np.asarray(v))
    
    def generate_weight(self, x):
        pass
    
    def generate_hessian(self, x):
        weights = np.diag(self.generate_weight(x))
        slack_inv = np.diag(1/self.generate_slack(x))
        
        matrix = np.matmul(np.matmul(slack_inv, weights), slack_inv)
        matrix = np.matmul(np.matmul(self.A.T, matrix), self.A)
        
        return matrix
    
    def generate_proposal_density(self, x, z):
        matrix = self.generate_hessian(x)
        return np.sqrt(np.linalg.det(matrix)) * np.exp(self.term_density * self.local_norm(x - z, matrix))
    
    def generate_sample(self, x):
        walk_hessian = sqrtm(np.linalg.inv(self.generate_hessian(x)))
        direction = self.generate_gaussian_rv()
        
        return x + self.term_sample * np.matmul(walk_hessian, direction)
    
    def generate_complete_walk(self, num_steps):
        results = np.zeros((num_steps, self.d))
        x = self.x
        for i in range(num_steps):
            z = self.generate_sample(x)
            if self.accept_reject(z):
                g_x_z = self.generate_proposal_density(x, z)
                g_z_x = self.generate_proposal_density(z, x)
                alpha = min(1, g_z_x/g_x_z)
                x = z if np.random.random() < alpha else x
            results[i] = x.T
        return results