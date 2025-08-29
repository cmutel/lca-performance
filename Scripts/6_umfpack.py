from scipy.sparse.linalg import factorized, spsolve
import numpy as np
import bw2calc as bc
import bw2data as bd
from pyinstrument import Profiler
from scipy import sparse
from pathlib import Path

bd.projects.set_current("ecoinvent-3.11-cutoff")

db = bd.Database("ecoinvent-3.11-cutoff")

fus = [db.random() for _ in range(500)]

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

lca.technosphere_matrix = lca.technosphere_matrix.tocsc()
lca.solver = factorized(lca.technosphere_matrix)

profiler = Profiler()
profiler.start()

for node in fus:
    lca.lci(demand=demand)
    lca.lcia_calculation()

profiler.stop()

with open(Path(__file__).parent.parent / "Results" / "multiple-full-calcs-umfpack.html", "w") as f:
    f.write(profiler.output_html())

profiler = Profiler()
profiler.start()

for node in fus:
    lca.build_demand_array({node.id: 1})
    supply_array = lca.solve_linear_system()
    supply_array_one_column = sparse.coo_matrix(
        (supply_array, (np.arange(supply_array.shape[0]), np.zeros(supply_array.shape[0]))),
        shape=(len(lca.dicts.activity), 1),
    )
    inventory_matrix_one_column = lca.biosphere_matrix @ supply_array_one_column
    score = (lca.characterization_matrix @ inventory_matrix_one_column).sum()

profiler.stop()

with open(Path(__file__).parent.parent / "Results" / "multiple-calcs-only-score-umfpack.html", "w") as f:
    f.write(profiler.output_html())

profiler = Profiler()
profiler.start()

characterized_inventory = lca.characterization_matrix @ lca.biosphere_matrix

for node in fus:
    lca.build_demand_array({node.id: 1})
    supply_array = lca.solve_linear_system()
    supply_array_one_column = sparse.coo_matrix(
        (supply_array, (np.arange(supply_array.shape[0]), np.zeros(supply_array.shape[0]))),
        shape=(len(lca.dicts.activity), 1),
    )
    score = (characterized_inventory @ supply_array_one_column).sum()

profiler.stop()

with open(Path(__file__).parent.parent / "Results" / "pre-characterize-umfpack.html", "w") as f:
    f.write(profiler.output_html())

# This is crazy slow, like 60 second for a chunk of 5. Dead end.
# profiler.start()

# characterized_inventory = lca.characterization_matrix @ lca.biosphere_matrix

# for i in range(0, len(fus), 100):
#     i = 0
#     chunk = fus[i:i + 100]

#     demand_array = np.zeros((len(lca.dicts.product), len(chunk)))

#     for i, node in enumerate(chunk):
#         demand_array[lca.dicts.product[node.id], i] = 1

#     supply_array = spsolve(lca.technosphere_matrix, demand_array)
#     supply_matrix = sparse.csc_matrix(supply_array)
#     score = (characterized_inventory @ supply_matrix).sum(axis=0)

# profiler.stop()

# with open(Path(__file__).parent.parent / "Results" / "chunked-umfpack.html", "w") as f:
#     f.write(profiler.output_html())
