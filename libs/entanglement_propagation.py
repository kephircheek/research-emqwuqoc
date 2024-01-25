import itertools
import math
from dataclasses import dataclass

import bec
import joblib
import numpy as np
from tqdm import tqdm


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
    if None in (p, k, n):
        raise TypeError("expected integer, not 'None'")
    return sum(
        p_state_scalar_lm(l, m, p, k, n)
        for l, m in itertools.product(range(k + 1), range(n - k + 1))
        if p == (l + m)
    )


def k_state(i: int, k: int, m: int, n: int):
    return bec.fock_state_constructor(bec.BEC_Qubits.init_default(n, 0), m, i=i, k=k)


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
    return sum(
        f_state_even_k_coeff(t, p, k, m, n) * tensor(fock(k[0], n), fock(k[m - 1], n))
        for k in k_sets
    ) / norm


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
    return sum(
        f_state_odd_k_coeff(t, p, k, m, n) * tensor(fock(k[0], n), fock(k[m - 1], n))
        for k in k_sets
    ) / norm


def f_state_odd_k_coeff(t: float, p: int, k: tuple[int], m: int, n: int):
    return (
        math.sqrt(math.prod(math.comb(n, k[i]) for i in range(0, m, 2)))
        * math.prod(omega(t, j, k, n) for j in range(1, m - 1, 2))
        * math.prod(p_state_scalar(p, k[i], n) for i in range(2, m - 2, 2))
    )


def rho_b(t: float, p: int, k: tuple[int], m: int, n: int) -> list[list[complex]]:
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

    return rho.tolist()


def rho_b_ent(t: float, p: int, k: tuple[int], m: int, n: int):
    r = rho_b(t, p, k, m, n)
    return entropy_vn(r)


def entropy_vn(m, base=2):
    if base != 2:
        raise ValueError("invalid base: {base} != 2")
    eigvals = [v.real for v in np.linalg.eigvals(m)]
    eigvals_sum = sum(eigvals)
    return -sum(
        l / eigvals_sum * math.log2(l / eigvals_sum)
        for l in eigvals
        if eigvals_sum > 1e-13 and l > 1e-13
    )


@dataclass(frozen=True)
class PropagateEntanglementTask:
    n_bosons: int
    n_sites: int
    t_span: tuple[float]
    k_measured: list[int] = None
    projection: int = None

    def __post_init__(self):
        if self.k_measured is None:
            object.__setattr__(
                self,
                "k_measured",
                (self.n_bosons,) * (self.n_sites // 2 - ((self.n_sites + 1) % 2)),
            )
        if self.projection is None and self.n_sites > 3:
            object.__setattr__(self, "projection", self.n_bosons)

    @property
    def t_list(self):
        return list(np.linspace(*self.t_span))

    @property
    def label(self):
        return (
            f"n{self.n_bosons}m{self.n_sites}"
            + (
                f"k{','.join(map(str, self.k_measured))}"
                if len(self.k_measured) > 0
                else ""
            )
            + (f"p{self.projection}" if self.projection is not None else "")
        )

    def run(self, verbose=True, n_jobs=-2):
        states = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(rho_b)(
                t,
                p=self.projection,
                k=self.k_measured,
                m=self.n_sites,
                n=self.n_bosons,
            )
            for t in tqdm(self.t_list, postfix=self.label, disable=not verbose, ncols=80)
        )
        return PropagateEntanglementResult(task=self, t_list=self.t_list, states=states)


@dataclass(frozen=True)
class PropagateEntanglementResult:
    task: PropagateEntanglementTask
    t_list: list[float]
    states: list[list[complex]]

    def entropies(self, verbose=False):
        return [entropy_vn(s) for s in tqdm(self.states, disable=not verbose)]

    def reveal_state(self, indx: int):
        rho = 0
        model = bec.BEC_Qubits.init_default(self.task.n_bosons, 0)
        state = self.states[indx]
        for km1, km2 in itertools.combinations_with_replacement(
            range(self.task.n_bosons + 1), 2
        ):
            elem = state[km1][km2]
            km1_vec = bec.fock_state_constructor(
                model, n=1, i=0, k=km1
            ) * bec.vacuum_state(model, n=1)
            km2_vec = bec.fock_state_constructor(
                model, n=1, i=0, k=km2
            ) * bec.vacuum_state(model, n=1)
            rho += km1_vec * km2_vec.dag() * elem
            if km1 != km2:
                rho += km2_vec * km1_vec.dag() * elem.conjugate()
        return rho / 2**self.task.n_bosons  # `... / 2^n` added for Tr{rho} == 1
