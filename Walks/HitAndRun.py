from RandomWalk import *

class HitAndRun(RandomWalk):
    def __init__(self, A, b, r, err):
        super().__init__(A, b, r)
        self.error = err
    
    def distance(self, x, y):
        return np.linalg.norm(x - y)
    
    def binary_search(self, direction, x):
        if self.r == 0:
            return
        farthest = x + self.r * direction
        dist = 0

        while True:
            dist = self.distance(x, farthest)
            farthest = x + 2 * dist * direction
            if not self.accept_reject(farthest):
                break
        left = x
        right = farthest
        mid = (left + right)/2

        while self.distance(left, right) > self.error:
            mid = (left + right)/2
            if self.accept_reject(mid):
                left = mid
            else:
                right = mid
        return self.distance(mid, x)
    
    def random_sample(self, x, num_sim):
        n = len(x)
        results = np.zeros((num_sim, n))
        for i in range(num_sim):
            new_direct = self.generate_n_rv(n)
            pos_side = self.binary_search(new_direct, x)
            neg_side = self.binary_search(new_direct * -1, x) * -1
            random_point = np.random.random() * (pos_side - neg_side) + neg_side
            x = random_point * new_direct + x
            results[i] = x.T
        return results  