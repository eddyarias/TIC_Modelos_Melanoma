import numpy as np
from visdom import Visdom

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, y_label, var_legend, title_name, x, y):
        if y_label not in self.plots:
            self.plots[y_label] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[var_legend],
                title=title_name,
                xlabel='Epochs',
                ylabel=y_label
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[y_label], name=var_legend, update = 'append')
