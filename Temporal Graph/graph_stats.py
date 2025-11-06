# tiny snippet you can run once
import collections
import glob
import json
from collections import Counter

import dgl

gs = []
for p in glob.glob("Temporal Graph/processed/dgl_temporal/*.dgl"):
    g, _ = dgl.load_graphs(p)
    gs += g
print("graphs:", len(gs))
print("avg nodes:", sum(g.num_nodes() for g in gs) / len(gs))
print("avg edges:", sum(g.num_edges() for g in gs) / len(gs))
print("% with 0 entity edges:", sum(int((g.edata["e_type"] == 1).sum() == 0) for g in gs) / len(gs) * 100)

meta = json.load(open("Temporal Graph/processed/dgl_temporal/index.json"))
cnt = collections.Counter([m["y"] for m in meta])
print(cnt)
