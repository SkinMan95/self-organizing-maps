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
        ax.scatter(x[0], x[1], label='input')
        ax.scatter(y[0], y[1], label='output', color='red')
        
        plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    import math
    x = np.arange(0, 10, 0.05)
    y = np.sin(x)
    x = x
    yy = np.vstack((x,y)).transpose().tolist()
    
    logging.debug("x: \n%s", x)
    logging.debug("yy: \n%s", yy)

    it = 2**10
    lr = 0.1
    dim = 100
    tau1 = 1000
    tau2 = 1000/math.log(dim)
    
    som = SOM(it, lr, dim, tau1, tau2, None, bidimensional=False)
    r = som.solve(yy)
    logging.info("r: \n%s", r)

    gui = SOMGUI()
    gui.plot(np.array(yy), r)
