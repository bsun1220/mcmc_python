nfrom RandomWalk import *

class BallWalk(RandomWalk):
    def __init__(self, A, b, r):
        super().__init__(A, b, r)
    
    def random_sample(self, x, num_sim):
        n = len(x)
        results = np.zeros((num_sim, n))
        for i in range(num_sim):
            new_x = self.generate_n_rv(n) * self.r + x
            if self.accept_reject(new_x):
                x = new_x
            results[i] = x.T
        return results