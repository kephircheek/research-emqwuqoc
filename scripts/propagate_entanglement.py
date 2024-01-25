import itertools
import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent / "libs"))

from dataclasses import asdict
import math

from entanglement_propagation import PropagateEntanglementTask, PropagateEntanglementResult


force = False

n_jobs = -2
n_bosons = 15
# t_span = (0, math.pi / 4, 301)
t_span = (0.35, 0.45, 201)

numbers_sites = [5]
k_measured_sets = [
    (n_bosons, n_bosons),
    # (n_bosons, 0),
    # (0, 0),
    # (n_bosons - 1, n_bosons - 1),
    # (n_bosons - 1, n_bosons),
] + [(random.randint(0, n_bosons), random.randint(0, n_bosons)) for _ in range(5)]
# k_measured_sets = [(n_bosons, n_bosons)]
# projections = [0, n_bosons - 1, n_bosons]
# projections = list(range(0, n_bosons + 1, 3))
projections = [n_bosons, n_bosons - 1, n_bosons - 2, 0, 1, 2]
verbose = True

file = pathlib.Path(__file__)
results_path = file.parent.parent / "assets" / "scripts" / file.stem / "results"
results_path.mkdir(parents=True, exist_ok=True)
print(f"Dir of results: '{results_path}'")

cases = list(itertools.product(numbers_sites, k_measured_sets, projections))
for i, (n_sites, k_measured, projection) in enumerate(cases):
    task = PropagateEntanglementTask(
        n_bosons=n_bosons,
        n_sites=n_sites,
        t_span=t_span,
        k_measured=k_measured,
        projection=projection,
    )
    result_filename = f"{task.label}.json"
    if (results_path / result_filename).exists():
        if not force:
            print(f"File '{result_filename}' already exist! Skipped... {i}/{len(cases)}")
            continue
        print(f"File '{result_filename}' already exist! It will be overwritten")

    result = task.run(n_jobs=n_jobs, verbose=verbose, ncols=180)
    print(f"Saving '{result_filename}' ...", end=" ")
    with open(results_path / result_filename, "w") as f:
        result.dump(f)
    print(f"OK {i}/{len(cases)}")
