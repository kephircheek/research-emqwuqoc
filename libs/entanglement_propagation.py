from dataclasses import dataclass
import itertools
import math

import bec
import numpy as np


def xi(j: int, k: tuple[int], n: int):
    """See comment in Eq.7."""
    return n - 2 * k[j]


def omega(t: float, j: int, k: tuple[int], n: int):
    """See Eq.13."""
    return (
        math.sqrt(math.comb(n, k[j]))
        * math.cos((xi(j - 1, k, n) + xi(j + 1, k, n)) * t) ** k[j]
        * math.sin((xi(j - 1, k, n) + xi(j + 1, k, n)) * t) ** (n - k[j])
    )


def p_state_scalar_lm(l: int, m: int, p: int, k: int, n: int):
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
        k: measured values
        p: some project state number
        m: number of qubits in chain
        n: number of bosons in qubit
    """
    if m % 2 == 0:
        return f_state_even(t, p, k, m, n)
    return f_state_odd(t, p, k, m, n)


def f_state_even(*args, **kwargs):
    if len(k) < (m // 2 - 1):
        raise ValueError("too few measured sites")
    norm = 2 ** (m * n / 4)
    k_sets = list(
        itertools.product(
            *itertools.chain(
                *itertools.zip_longest(
                    (
                        range(n + 1) for _ in range(0, m // 2)
                    ),  # since last range added below)
                    ([k_] for k_ in k),
                ),
                [range(n + 1)],
                [range(n + 1)],
            )
        )
    )
    return sum(f_state_even_k(t, p, k, m, n) for k in k_sets) / norm


def f_state_even_k(t: float, p: int, k: tuple[int], m: int, n: int):
    return (
        f_state_even_k_coeff(t, p, k, m, n)
        * k_state(0, k[0], 2, n)
        * k_state(1, k[m - 1], 2, n)
        * bec.vacuum_state(bec.BEC_Qubits.init_default(n, 0), n=2)
    )


def f_state_even_k_coeff(t: float, p: int, k: tuple[int], m: int, n: int):
    return (
        math.sqrt(
            math.prod(math.comb(n, k[i]) for i in range(0, m - 1, 2))
            * math.comb(n, k[-1])
        )
        * math.prod(omega(t, j, k, n) for j in range(1, m - 2, 2))
        * math.prod(p_state_scalar(p, k[i], n) for i in range(2, m - 1, 2))
        * (np.exp(1j * xi(m - 2, k, n) * t) / math.sqrt(2)) ** k[-1]
        * (np.exp(-1j * xi(m - 2, k, n) * t) / math.sqrt(2)) ** (n - k[-1])
    )


def f_state_odd(t: float, p: int, k: tuple[int], m: int, n: int):
    if len(k) < m // 2:
        raise ValueError("too few measured sites")
    norm = 2 ** ((m + 1) * n / 4)
    k_sets = list(
        itertools.product(
            *itertools.chain(
                *itertools.zip_longest(
                    (
                        range(n + 1) for _ in range(0, m // 2)
                    ),  # since last range added below)
                    ([k_] for k_ in k),
                ),
                [range(n + 1)],
            )
        )
    )
    return sum(f_state_odd_k(t, p, k, m, n) for k in k_sets) / norm


def f_state_odd_k_coeff(t: float, p: int, k: tuple[int], m: int, n: int):
    return (
        math.sqrt(math.prod(math.comb(n, k[i]) for i in range(0, m, 2)))
        * math.prod(omega(t, j, k, n) for j in range(1, m - 1, 2))
        * math.prod(p_state_scalar(p, k[i], n) for i in range(2, m - 2, 2))
    )


def f_state_odd_k(t: float, p: int, k: tuple[int], m: int, n: int):
    return (
        f_state_odd_k_coeff(t, p, k, m, n)
        * k_state(0, k[0], 2, n)
        * k_state(1, k[m - 1], 2, n)
        * bec.vacuum_state(bec.BEC_Qubits.init_default(n, 0), n=2)
    )


def rho_b(t: float, p: int, k: tuple[int], m: int, n: int):
    rho = np.zeros((n + 1, n + 1), dtype=complex)
    for km1, km2 in itertools.combinations_with_replacement(range(n + 1), 2):

        def k_sets_odd(k0, km):
            return itertools.product(
                *itertools.chain(
                    [[k0]],
                    *itertools.zip_longest(
                        ([k_] for k_ in k[:-1]),
                        (range(n + 1) for _ in range(0, m // 2 - 1)),
                    ),
                    [[k[-1]]],
                    [[km]],
                )
            )

        def k_sets_even(k0, km):
            return itertools.product(
                *itertools.chain(
                    [[k0]],
                    *itertools.zip_longest(
                        ([k_] for k_ in k),
                        (range(n + 1) for _ in range(0, m // 2 - 1)),
                    ),
                    [[km]],
                )
            )

        if m % 2 == 0:
            f_state_k_coeff = f_state_even_k_coeff
            k_sets = k_sets_even
        else:
            f_state_k_coeff = f_state_odd_k_coeff
            k_sets = k_sets_odd

        elem = sum(
            sum(f_state_k_coeff(t, p, k_, m, n) for k_ in k_sets(k0, km1))
            * sum(f_state_k_coeff(t, p, k_, m, n) for k_ in k_sets(k0, km2)).conjugate()
            for k0 in range(n + 1)
        )

        rho[km1, km2] = elem
        if km1 != km2:
            rho[km2, km1] = elem.conjugate()

    return rho


def rho_b_ent(t: float, p: int, k: tuple[int], m: int, n: int):
    r = rho_b(t, p, k, m, n)
    eigvals = [e.real for e in np.linalg.eigvals(r)]
    eigvals_sum = sum(eigvals)
    return -sum(
        l / eigvals_sum * math.log2(l / eigvals_sum)
        for l in eigvals
        if eigvals_sum > 1e-13 and l > 1e-13
    )
