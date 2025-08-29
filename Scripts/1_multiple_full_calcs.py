import bw2calc as bc
import bw2data as bd
from pyinstrument import Profiler
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

profiler = Profiler()
profiler.start()

for node in fus:
    lca.lci(demand={node.id: 1})
    lca.lcia_calculation()

profiler.stop()

with open(Path(__file__).parent.parent / "Results" / "multiple-full-calcs.html", "w") as f:
    f.write(profiler.output_html())
