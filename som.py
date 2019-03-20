import numpy as np
import math
import random
import logging

class SOM(object):

    logger = logging.getLogger('SOM')
    learning_threshold = 0.0001
    
    def __init__(self, iterations, learning_rate, dimensions, tau1, tau2, kernelfunc, bidimensional=False):
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
        

    def gen_neurons(self,n):
        """Generates the neurons positions and weights"""
        assert isinstance(n, int) and n > 0
        SOM.logger.debug("Creating %d neurons positions and weights", n)
        W = np.random.uniform(size=self.dimensions + (n,))
        SOM.logger.debug("Initial weight: \n%s", W)
        
        O = np.array(list(np.ndindex(self.dimensions)))

        return W, O

    # TODO: euclidean distance for now
    def distance(self, a, b=None):
        if b is None:
            return np.linalg.norm(a)
        return np.linalg.norm(a - b)
        
    def winner(self, W, O, x) -> int:
        """Winner neuron for input x"""
        mn, mn_dist = None, math.inf
        for j in range(len(O)):
            d = self.distance(x, W[O[j]])
            # SOM.logger.debug("neuron %d, weight: %s, position: %s, input: %s, minimum: %s, distance: %f", j, W[O[j]], O[j], x, mn, d)
            if d < mn_dist:
                mn, mn_dist = j, d

        return mn

    def learning_rate_func(self, t):
        assert t >= 0
        val = self.learning_rate * np.exp(-t/self.tau1)
        return max(val, SOM.learning_threshold)

    def neigh_function(self, t):
        assert t >= 0
        return max(self.dimensions[0] * np.exp(-t/self.tau2), 0.1)

    def neighborhood_function(self, t, O, winner):
        w = O[winner]
        diff = np.square(np.linalg.norm(O - np.ones(O.shape) * w, axis=1))
        return np.exp( - diff / (2 * self.neigh_function(t)**2) )

    def weights_diff(self, t, neighfunc, W, x):
        dim = W.shape[0]
        # SOM.logger.debug("weights dims: %d", dim)
        diff = np.tile(x, (dim,1)) - W
        neighfunc = np.reshape(neighfunc, self.dimensions)
        neighfunc = np.tile(neighfunc, (diff.shape[-1],) + (1,) * neighfunc.ndim).transpose()
        # SOM.logger.debug("neighfunc: \n%s", neighfunc)
        # SOM.logger.debug("weights_diff before:\n%s", neighfunc * diff)
        
        r = self.learning_rate_func(t) * neighfunc * diff
        # SOM.logger.debug("weights_diff after:\n%s", r)
        return r
        
    def solve(self, x):
        assert isinstance(x, (list, np.ndarray)) and len(x) > 0
        n = len(x[0])
        assert all([len(xi) == n for xi in x]), "all vectors of input should be of the same dimensions"
        assert all([all([isinstance(xi, (int,float)) for xi in x[i]]) for i in range(n)]), "every element should be either an int or a float"

        x = np.array(x, dtype=np.float64)
        SOM.logger.debug("input pattern: \n%s", x)
        W, O = self.gen_neurons(n)

        for t in range(1, self.iterations+1):
            vet_permut = list(range(len(x)))
            random.shuffle(vet_permut)
            # SOM.logger.debug("vet_permut: <%s>", vet_permut)
            for j in range(len(vet_permut)):
                i = vet_permut[j]
                # SOM.logger.debug("actual input %d <%s>", i, x[i])
                winner = self.winner(W, O, x[i])
                # SOM.logger.debug("winner(t: %d): %d", t, winner)
                
                hjix = self.neighborhood_function(t, O, winner)
                
                # SOM.logger.debug("neighborhood_function:\n%s", hjix)
                
                W += self.weights_diff(t, hjix, W, x[i])

            if t % (self.iterations // 10) == 0:
                SOM.logger.info("progress (%d%%) <t: %d, learning_rate: %f>", 100 * t / self.iterations, t, self.learning_rate_func(t))

        return W
                
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    import math
    with open("data/iris.csv", 'r') as f:
        x = [[float(i) for i in line.strip().split(",")] for line in f.readlines()]
        dim = 12
        tau = 15
        som = SOM(20, 0.05, dim, tau, tau/math.log(dim), None, bidimensional=False)
        print(som.solve(x))

