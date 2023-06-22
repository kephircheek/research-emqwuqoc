import matplotlib.pyplot as plt
import numpy as np
import qutip
from matplotlib import animation


def animate_state_evolution_on_bloch_sphere(evolution, psi_target):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    sphere = qutip.Bloch(axes=ax)

    def animate(i):
        sphere.add_states(evolution.states[i])
        sphere.make_sphere()
        return ax

    def init():
        sphere.add_states(psi_target)
        sphere.vector_color = ["r"] + ["grey"] * len(evolution.states)
        return ax

    ani = animation.FuncAnimation(
        fig,
        animate,
        np.arange(len(evolution.states)),
        init_func=init,
        blit=False,
        repeat=False,
    )

    return ani
