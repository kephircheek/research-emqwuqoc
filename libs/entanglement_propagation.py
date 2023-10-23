import itertools
import math

import bec
import numpy as np


def xi(j: int, k: tuple[int], n: int):
    """See comment in Eq.7."""
    return n - 2 * k[j]


def omega(t: float, j: int, k: tuple[int], n: int):
    """See Eq.13."""
    # print((j, math.comb(n, k[j])))
    return (
        math.sqrt(math.comb(n, k[j]))
        * math.cos((xi(j - 1, k, n) + xi(j + 1, k, n)) * t) ** k[j]
        * math.sin((xi(j - 1, k, n) + xi(j + 1, k, n)) * t) ** (n - k[j])
    )


def p_state_scalar_lm(l: int, m: int, p: int, k: int, n: int):
    # print(math.factorial(l + m) * math.factorial(n - l - m))
    return (
        (-1) ** (n - k - m)
        * math.comb(k, l)
        * math.comb(n - k, m)
        * math.sqrt(
            math.factorial(l + m)
            * math.factorial(n - l - m)
            / math.factorial(k)
            / math.factorial(n - k)
            / 2**n
        )
    )


def p_state_scalar(p: int, k: int, n: int):
    return sum(
        p_state_scalar_lm(l, m, p, k, n)
        for l, m in itertools.product(range(k + 1), range(n - k + 1))
        if p == (l + m)
    )


def k_state(i: int, k: int, m: int, n: int):
    return bec.fock_state_constructor(bec.BEC_Qubits.init_default(n, 0), m, k=i, i=k)


def f_state(t: float, p: int, k: tuple[int], m: int, n: int):
    """
    Return final state, see eq. 11 and eq. 12.

    Args:
        t: time of evolution
        k: mesu
        p: some project state number
        m: number of qubits in chain
        n: number of bosons in qubit
    """
    if m % 2 == 0:
        return f_state_even(t, p, k, m, n)
    return f_state_odd(t, p, k, m, n)


def f_state_even(*args, **kwargs):
    raise NotImplementedError


def f_state_odd(t: float, p: int, k: tuple[int], m: int, n: int):
    if len(k) < m // 2:
        raise ValueError("too few measured sites")
    norm = 2 ** ((m + 1) * n / 4)
    k_sets = list(
        itertools.product(
            *itertools.chain(
                *itertools.zip_longest(
                    (
                        range(n + 1) for _ in range(0, m - 1, 2)
                    ),  # since last range added below)
                    ([k] for k in k),
                ),
                [range(n + 1)]
            )
        )
    )
    return sum(f_state_odd_k(t, p, k, m, n) for k in k_sets) / norm
    # return sum(abs(f_state_odd_k(t, p, k, m, n))**2 for k in k_sets) / norm / norm_


def f_state_odd_k(t: float, p: int, k: tuple[int], m: int, n: int):
    # print("k", k)
    # print(list((i, math.comb(n, k[i])) for i in range(0, m, 2)))
    # print("<p|ki>", list(p_state_scalar(p, k[i], n) for i in range(2, m - 2, 2)))
    # return math.prod(math.comb(n, ki) for ki in k) \
    # return math.sqrt(math.prod(math.comb(n, k[i]) for i in range(0, m, 2))) \
    result = (
        math.sqrt(math.prod(math.comb(n, k[i]) for i in range(0, m, 2)))
        * math.prod(omega(t, j, k, n) for j in range(1, m - 1, 2))
        * math.prod(p_state_scalar(p, k[i], n) for i in range(2, m - 2, 2))
        * k_state(0, k[0], 2, n)
        * k_state(1, k[m - 1], 2, n)
        * bec.vacuum_state(bec.BEC_Qubits.init_default(n, 0), n=2)
    )
    return result


def rhob(s, n):
    rho = s * s.dag

    def k(i):
        return k_state(0, i, 2, n) * bec.vacuum_state(
            bec.BEC_Qubits.init_default(n, 0), n=2
        )

    rhob_k = sum(k(i) for i in range(n + 1))


if __name__ == "__main__":
    pass
    n = 10
    m = 3
    k_measured = (0,) * (m // 2)
    p = 1
    tspan = np.linspace(0, 1, 3)
    states = [f_state(t, p, k_measured, m, n) for t in tspan]
    # print(states)
    # exit()
    print(*(((s.dag() * s).full()[0, 0]).real for s in states))

    # import qutip

    # @np.vectorize
    # def entropy_vn(t):
    #     s = f_state(t, p, k_measured, m, n)
    #     return qutip.entropy.entropy_vn(qutip.ptrace(s, [0, 1]))

    # tspan = np.linspace(0, 1.5, 150)
    # entropy_vn_span = entropy_vn(tspan)
    # import matplotlib.pyplot as plt
    # plt.plot(tspan, entropy_vn_span, label=f"p{p}m{m}n{n}k{''.join(map(str, k_measured))}")
    # plt.legend()
    # plt.show()
