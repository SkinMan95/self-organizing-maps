import matplotlib.pyplot as plt
import numpy as np
import logging
from som import SOM

class Plot(object):

    logger = logging.getLogger('Plot')
    
    def __init__(self):
        Plot.logger.debug('new Plot instance')

    def plot_data(self, ax, x): # extensibility for more complex graphs
        ax.plot(x)
        

class SOMGUI(object):

    logger = logging.getLogger('SOMGUI')

    def __init__(self, plotfunc=None):
        self.plotfunc = plotfunc if plotfunc is not None else Plot()
        Plot.logger.debug('new SOMGUI instance')

    def plot(self, x, y):
        fig, ax = plt.subplots()
        x = x.transpose()
        y = y.transpose()
        ax.scatter(x[0], x[1])
        ax.plot(y[0], y[1], 'rx-')
        
        plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    import math

    delta = np.pi / 2.0 / 24.0
    
    logging.debug("delta: %f", delta)
    
    theta = np.arange(0, 2.0 * np.pi * 1.5 , delta)
    logging.debug("input size: %s", theta.shape)
    
    a = 1.2
    b = 0.3
    r = a * np.exp(b * theta)
    var = 0.5
    x = r * np.cos(theta) + np.random.uniform(low=-var, high=var, size=theta.shape)
    y = r * np.sin(theta) + np.random.uniform(low=-var, high=var, size=theta.shape)
    yy = np.vstack((x,y)).transpose().tolist()
    
    logging.debug("x: \n%s\ny:\n%s", x, y)
    logging.debug("yy: \n%s", yy)

    it = 2**10
    lr = 0.1
    dim = 50
    tau1 = 1000
    tau2 = 1000/math.log(dim)
    
    som = SOM(it, lr, dim, tau1, tau2, None, bidimensional=False)
    r = som.solve(yy)
    logging.info("r: \n%s", r)

    gui = SOMGUI()
    gui.plot(np.array(yy), r)
