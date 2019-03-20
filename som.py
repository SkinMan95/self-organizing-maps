import numpy as np
import random
import logging

class SOM(object):

    logger = logging.getLogger('SOM')
    
    def __init__(self, iterations, learning_rate, dimensions, tau1, tau2, kernelfunc, randfunc=None, bidimensional=False):
        self.iterations = iterations
        assert isinstance(iterations, int)
        assert iterations > 0
        
        self.learning_rate = learning_rate
        assert isinstance(learning_rate, float)
        assert 0.0 < learning_rate < 1.0, "learning rate should be domain (0, 1)"
        
        assert isinstance(dimensions, int)
        assert dimensions > 0
        assert isinstance(bidimensional, bool)
        self.bidimensional = bidimensional
        if self.bidimensional:
            self.dimensions = (dimensions, dimensions)
        else:
            self.dimensions = (dimensions,)
            
        
        self.tau1 = tau1
        assert isinstance(tau1, (int,float))
        assert tau1 > 0
        
        self.tau2 = tau2
        assert isinstance(tau2, (int,float))
        assert tau2 > 0

        # TODO: use it in the neighborhood function
        self.kernelfunc = kernelfunc # should be a lambda function of one argument

        self.randfunc = randfunc if randfunc is not None else np.random.random
        

    def gen_neurons(self,n):
        """Generates the neurons positions and weights"""
        assert isinstance(n, int) and n > 0
        SOM.logger.debug("Creating %d neurons positions and weights", n)
        W = self.randfunc(self.dimensions + (n,))
        O = np.array(list(np.ndindex(self.dimensions)))

        return W, O

    def distance(self, a, b=None): # TODO: euclidean distance for now
        if b is None:
            return np.linalg.norm(a)
        return np.linalg.norm(a - b)
        
    def winner(self, W, O, x) -> int:
        """Winner neuron for input x"""
        mn, mn_dist = None, float('inf')
        for j in range(len(O)):
            d = self.distance(W[O[j]], x)
            if d < mn_dist:
                mn, mn_dist = j, d

        return mn

    def learning_rate_func(self, t):
        assert t >= 0
        val = self.learning_rate * np.exp(-t/self.tau1)
        assert 0.0 < val < 1.0
        return val

    def neigh_function(self, t):
        assert t >= 0
        return self.dimensions[0] * np.exp(-t/self.tau2)

    def neighborhood_function(self, t, O, winner):
        w = O[winner]
        diff = np.square(np.linalg.norm(O - np.ones(O.shape) * w, axis=1))
        return np.exp( - diff / (self.neigh_function(t)**2) * 2 )

    def weights_diff(self, t, neighfunc, W, x):
        dim = W.shape[0]
        SOM.logger.debug("weights dims: %d", dim)
        diff = np.tile(x, (dim,1)) - W
        neighfunc = np.reshape(neighfunc, self.dimensions)
        neighfunc = np.tile(neighfunc, (diff.shape[-1],) + (1,) * neighfunc.ndim).transpose()
        return self.learning_rate_func(t) * neighfunc * diff
        
    def solve(self, x):
        assert isinstance(x, list) and len(x) > 0
        n = len(x[0])
        assert all([len(xi) == n for xi in x]), "all vectors of input should be of the same dimensions"
        assert all([all([isinstance(xi, (int,float)) for xi in x[i]]) for i in range(n)]), "every element should be either an int or a float"

        x = np.array(x, dtype=np.float64)
        W, O = self.gen_neurons(n)

        for t in range(self.iterations):
            vet_permut = list(range(n))
            random.shuffle(vet_permut)
            for j in range(n):
                i = vet_permut[j]
                winner = self.winner(W, O, x[i])
                hjix = self.neighborhood_function(t, O, winner)
                W += self.weights_diff(t, hjix, W, x[i])

        return W
                
if __name__ == "__main__":
    import math
    with open("data/iris.csv", 'r') as f:
        x = [[float(i) for i in line.strip().split(",")] for line in f.readlines()]
        som = SOM(20, 0.1, 5, 1000, 1000/math.log(5), None, bidimensional=True)
        print(som.solve(x))

