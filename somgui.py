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
        ax.plot(x[0], x[1], label='input')
        ax.scatter(y[0], y[1], label='output')
        
        plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    import math
    dim = 10
    x = np.arange(0, dim, 0.2)
    y = np.sin(x)
    yy = np.vstack((x,y)).transpose().tolist()
    logging.debug("x: \n%s", x)
    logging.debug("yy: \n%s", yy)
    
    tau = 1.5 * dim
    
    som = SOM(50, 0.5, 50 , tau, tau/math.log(dim), None, bidimensional=False)
    r = som.solve(yy)
    logging.info("r: \n%s", r)

    gui = SOMGUI()
    gui.plot(np.array(yy), r)
