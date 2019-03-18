import numpy as np
import random
from utils import *

class SOM(object):
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
            self.dimensions = (dimensions)
            
        
        self.tau1 = tau1
        assert isinstance(tau1, (int,float))
        assert tau1 > 0
        
        self.tau2 = tau2
        assert isinstance(tau2, (int,float))
        assert tau2 > 0
        
        self.kernelfunc = kernelfunc # should be a lambda function of one argument

        self.randfunc = randfunc if randfunc is not None else np.random.random
        

    def gen_neurons(self,n):
        """Generates the neurons positions and weights"""
        assert isinstance(n, int) and n > 0
        W = self.randfunc(self.dimensions + (n,))
        O = np.array(list(np.ndindex(self.dimensions)))

        return W, O

    def distance(self, a, b): # TODO: euclidean distance for now
        return np.linalg.norm(a - b)
        
    def winner(self, W, O, x) -> int:
        """Winner neuron for input x"""
        mn, mn_dist = None, float('inf')
        for j in range(len(O)):
            d = self.distance(W[O[j]], x)
            if d < mn_dist:
                mn, mn_dist = j, d

        return mn
        
    def solve(self, x):
        assert isinstance(x, list) and len(x) > 0
        n = len(x[0])
        assert all([len(xi) == n for xi in x]), "all vectors of input should be of the same dimensions"
        assert all([all([isinstance(xi, (int,float)) for xi in x[i]]) for i in range(n)]), "every element should be either an int or a float"

        x = np.array(x, dtype=np.float64)
        W, O = self.gen_neurons(n)

        for t in range(self.iterations):
            vet_permut = random.shuffle(range(n))
            for j in range(n):
                i = vet_permut[j]
                ix = self.winner(W, O, x[i])
                Utilities().lprint(ix)
