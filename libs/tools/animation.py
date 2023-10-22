import matplotlib.pyplot as plt
import numpy as np
import qutip
from matplotlib import animation


class BlochAnimation:
    def __init__(self, states, target):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")
        self.sphere = qutip.Bloch(axes=self.ax)

        self.states = states
        self.target = target

    def _animate(self, i):
        self.sphere.add_states(self.states[i])
        self.sphere.make_sphere()
        return self.ax

    def _init(self):
        self.sphere.add_states(self.target)
        self.sphere.vector_color = ["red"] + ["blue"] + ["grey"] * len(self.states[1:])
        return self.ax

    def animation(self):
        ani = animation.FuncAnimation(
            self.fig,
            self._animate,
            np.arange(len(self.states)),
            init_func=self._init,
            blit=False,
            repeat=False,
        )
        return ani

    def plot_overlay(self):
        self._init()
        for i in range(len(self.states)):
            self._animate(i)
        return self.ax
