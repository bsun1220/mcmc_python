import numpy as np
import pandas as pd

class RandomWalk:
    def __init__(self, A, b, r):
        self.A = A
        self.b = b
        self.r = r
    
    def accept_reject(self, x):
        return (np.matmul(self.A,x) < self.b).all()
    
    def generate_n_rv(self, n):
        arr = np.random.randn(n)
        return (arr/np.linalg.norm(arr))[..., None]
    
    def random_sample(self):
        pass

