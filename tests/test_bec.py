import unittest
from dataclasses import replace

import qutip

from libs import bec


class TestOperators(unittest.TestCase):
    def setUp(self):
        self.model = bec.BEC_Qubits.init_default(3, 0)

    def test_a_of_single_qubit(self):
        dim = self.model.n_bosons + 1
        expected = qutip.tensor(qutip.destroy(dim), qutip.identity(dim))
        actual = bec.a(self.model)
        self.assertEqual(expected, actual)

    def test_b_of_single_qubit(self):
        dim = self.model.n_bosons + 1
        expected = qutip.tensor(qutip.identity(dim), qutip.destroy(dim))
        actual = bec.b(self.model)
        self.assertEqual(expected, actual)

    def test_a_of_single_qubit_with_excitation_level(self):
        model = replace(self.model, excitation_level=True)
        dim = model.n_bosons + 1
        expected = qutip.tensor(
            qutip.destroy(dim), qutip.identity(dim), qutip.identity(dim)
        )
        actual = bec.a(model)
        self.assertEqual(expected, actual)

    def test_b_of_single_qubit_with_excitation_level(self):
        model = replace(self.model, excitation_level=True)
        dim = model.n_bosons + 1
        expected = qutip.tensor(
            qutip.identity(dim), qutip.destroy(dim), qutip.identity(dim)
        )
        actual = bec.b(model)
        self.assertEqual(expected, actual)

    def test_e_of_single_qubit_with_excitation_level(self):
        model = replace(self.model, excitation_level=True)
        dim = model.n_bosons + 1
        expected = qutip.tensor(
            qutip.identity(dim), qutip.identity(dim), qutip.destroy(dim)
        )
        actual = bec.e(model)
        self.assertEqual(expected, actual)

    def test_a_of_first_qubit_in_pair(self):
        dim = self.model.n_bosons + 1
        expected = qutip.tensor(
            qutip.destroy(dim),
            qutip.identity(dim),
            qutip.identity(dim),
            qutip.identity(dim),
        )
        actual = bec.a(self.model, 2, k=0)
        self.assertEqual(expected, actual)

    def test_b_of_first_qubit_in_pair(self):
        dim = self.model.n_bosons + 1
        expected = qutip.tensor(
            qutip.identity(dim),
            qutip.destroy(dim),
            qutip.identity(dim),
            qutip.identity(dim),
        )
        actual = bec.b(self.model, 2, k=0)
        self.assertEqual(expected, actual)
