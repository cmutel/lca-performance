from pypardiso.pardiso_wrapper import PyPardisoSolver
import numpy as np
import bw2calc as bc
import bw2data as bd
from pyinstrument import Profiler
from scipy import sparse
from pathlib import Path

bd.projects.set_current("ecoinvent-3.11-cutoff")

db = bd.Database("ecoinvent-3.11-cutoff")

nodes = [node for node in db]
nodes.sort(key=lambda x:x.id)
fus = nodes[:2500]

m = (
    'ecoinvent-3.11',
    'ReCiPe 2016 v1.03, endpoint (H) no LT',
    'ecosystem quality no LT',
    'photochemical oxidant formation: terrestrial ecosystems no LT'
)

demand, data_objs, _ = bd.prepare_lca_inputs(demand={fus[0]: 1}, method=m, remapping=False)

lca = bc.LCA(demand, data_objs=data_objs)
lca.lci()
lca.lcia()

solver = PyPardisoSolver()
solver.factorize(lca.technosphere_matrix)

profiler = Profiler(interval=0.000025)
profiler.start()

characterized_inventory = np.asarray((lca.characterization_matrix @ lca.biosphere_matrix).sum(axis=0))

def calculate_score(x, y):
    return x @ y

for i in range(0, len(fus), 100):
    i = 0
    chunk = fus[i:i + 100]

    demand_array = np.zeros((len(lca.dicts.product), len(chunk)))

    for i, node in enumerate(chunk):
        demand_array[lca.dicts.product[node.id], i] = 1

    b = solver._check_b(lca.technosphere_matrix, demand_array)
    solver.set_phase(33)
    supply_array = solver._call_pardiso(lca.technosphere_matrix, b)
    calculate_score(characterized_inventory, supply_array)

profiler.stop()

with open(Path(__file__).parent.parent / "Results" / "array-multiplication.html", "w") as f:
    f.write(profiler.output_html())
