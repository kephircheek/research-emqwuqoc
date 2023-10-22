import pathlib, sys

sys.path.append(str(pathlib.Path(sys.path[0]) / "libs"))

from dataclasses import replace

import math
import numpy as np
import qutip
from qutip.control.pulseoptim import optimize_pulse
import matplotlib.pyplot as plt
from tqdm import tqdm

import bec
from tools.jupyter import print_model_info
from tools.qutip import TqdmProgressBar

n_bosons = 3
phase = 0  # np.pi / 4
# model = BEC_Qubits.init_alexey2003(n_bosons=n_bosons, phase=phase)
model = bec.BEC_Qubits.init_default(
    n_bosons=n_bosons,
    phase=phase,
    excitation_level=True,
)
ecmodel = replace(model, excitation_level=True, communication_line=True)


def h_drift(m, n=2):
    return bec.h_int(m, n=n) + m.delta_l * (
        bec.e(m, n=n, k=0).dag() * bec.e(m, n=n, k=0)
        + bec.e(m, n=n, k=1).dag() * bec.e(m, n=n, k=1)
    )


def h_control(m, i, n=2):
    return bec.e(m, n=n, k=i).dag() * bec.b(m, n=n, k=i) + bec.b(
        m, n=n, k=i
    ).dag() * bec.e(m, n=n, k=i)


psi_initial = (
    bec.coherent_state_constructor(ecmodel, n=2, k=0)
    * bec.coherent_state_constructor(ecmodel, n=2, k=1)
    * bec.vacuum_state(ecmodel, n=2)
)

H0 = h_drift(ecmodel)
H1 = h_control(ecmodel, 0)
H2 = h_control(ecmodel, 1)

nt = 300
t_total = 2 / model.Omega / 2  # 5

tspan, dt = np.linspace(0, t_total, nt, retstep=True)
evolution = qutip.mesolve(
    H0 + H1 + H2,
    psi_initial,
    tspan,
    progress_bar=TqdmProgressBar(),
    options=qutip.Options(nsteps=1e5),
)

entropy = [qutip.entropy_vn(qutip.ptrace(s, [0, 1])) for s in tqdm(evolution.states)]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.plot(tspan, entropy, "*-", label="entropy")

ax.set_ylabel("S")
ax.set_xlabel("t")
ax.set_title("Энтропия фон Неймана")
ax.grid()
ax.legend()
plt.savefig(f"../assets/entropy_vn-b{model.n_bosons}-qoc-pump.pdf")

psi_target = evolution.states[-1]

# Fidelity error target
fid_err_targ = 1e-2
# Maximum iterations for the optisation algorithm
max_iter = 100
# Maximum (elapsed) time allowed in seconds
max_wall_time = 180
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-10
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
init_pulse_type = "RND"

result = optimize_pulse(
    H0,
    [H1, H2],
    psi_initial,
    psi_target,
    num_tslots=nt,
    evo_time=t_total,
    fid_err_targ=fid_err_targ,
    max_iter=max_iter,
    max_wall_time=max_wall_time,
    min_grad=min_grad,
    init_pulse_type=init_pulse_type,
    # dyn_type='SYMPL',
    gen_stats=True,
)

print("Fidelity:", result.fidelity)


amps = np.vstack(([0, 0], result.final_amps))
_, ax = plt.subplots()
ax.plot(result.time, amps[:, 0], label=r"$\alpha(t)$")
ax.plot(result.time, amps[:, 1], label=r"$\beta(t)$")
ax.set_xlabel("t")
ax.legend()
ax.set_title(r"Импульсы $ H_0 + g1(t) H_1 + g2(t) H_2 $")

plt.savefig(f"../assets/parameters-b{model.n_bosons}-qoc-pump.pdf")
