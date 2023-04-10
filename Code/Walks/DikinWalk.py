from BarrierRandomWalk import *

class DikinWalk(BarrierRandomWalk):
    def __init__(self, A, b, x, r):
        super().__init__(A, b, x, r)
        constant = (r ** 2)/self.d
        self.term_density = (-0.5/constant)
        self.term_sample = np.sqrt(constant)
    
    def generate_weight(self, x):
        return np.ones(self.n)