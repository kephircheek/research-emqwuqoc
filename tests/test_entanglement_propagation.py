import math
import tempfile
import unittest
from dataclasses import asdict

from entanglement_propagation import (PropagateEntanglementResult,
                                      PropagateEntanglementTask)


class TestPropagateEntanglementResult(unittest.TestCase):

    def setUp(self):
        n_jobs = 1
        n_bosons = 5
        n_sites = 3
        t_span = [0, math.pi / 4, 100]
        task = PropagateEntanglementTask(
            n_bosons=n_bosons,
            n_sites=n_sites,
            t_span=t_span,
            k_measured=[5], # by default is a tuple, but tuple loads as list
        )
        self.result = task.run(n_jobs=n_jobs, verbose=False)

    def test_dump_to_dict_and_recovering(self):
        result_dict = asdict(self.result)
        result_recoverd = PropagateEntanglementResult.init_form_dict(result_dict)
        self.assertEqual(self.result, result_recoverd)
        result_recovered_dict = asdict(result_recoverd)
        self.assertDictEqual(result_dict, result_recovered_dict)

    def test_dump_to_json_and_recovering(self):
        tmp = tempfile.NamedTemporaryFile()
        with open(tmp.name, 'w') as f:
            self.result.dump(f)
        with open(tmp.name) as f:
            result_recoverd = PropagateEntanglementResult.load(f)
            self.assertEqual(self.result, result_recoverd)
