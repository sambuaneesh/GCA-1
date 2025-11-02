# set the paths you described
data_path = "/home/stealthspectre/iiith/GCA/Extract triples/processed/out_supports_wiki.json"
save_path = "/home/stealthspectre/iiith/GCA/Extract triples/processed/graphs_wiki.json"

import json
import re


def parse_triple_str(s):
    s = s.strip()
    if s.lower().startswith("triple:"):
        s = s.split(":", 1)[1].strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",")]
    if len(parts) < 3:
        raise ValueError(f"bad triple: {s}")
    h, r, t = parts[0], parts[1], ",".join(parts[2:]).strip()
    return h, r, t


def triples_from_list(objs):
    nodes, rels, graph = set(), set(), []
    for o in objs:
        s = o["triple"] if isinstance(o, dict) and "triple" in o else str(o)
        try:
            h, r, t = parse_triple_str(s)
            nodes.update([h, t])
            rels.add(r)
            graph.append((h, r, t))
        except Exception:
            pass
    return list(nodes), list(rels), graph


def triples_from_block(block: str):
    nodes, rels, graph = set(), set(), []
    for line in block.splitlines():
        if "(" not in line:
            continue
        if "Triple:" in line:
            line = line.split(":", 1)[1]
        line = line.strip()
        try:
            h, r, t = parse_triple_str(line)
            nodes.update([h, t])
            rels.add(r)
            graph.append((h, r, t))
        except Exception:
            pass
    return list(nodes), list(rels), graph


data = json.load(open(data_path, "r", encoding="utf-8"))
if isinstance(data, dict):
    # allow dict-of-splits too
    entries = []
    for v in data.values():
        if isinstance(v, list):
            entries.extend(v)
else:
    entries = data

out = []
for e in entries:
    rec = {"entity": e.get("entity", ""), "label": e.get("label"), "sample0": {}}

    # Choose triples for sample0 in this order: fact_triples > support-only > all
    if "fact_triples" in e and e["fact_triples"]:
        nodes, rels, graph = triples_from_list([{"triple": t} for t in e["fact_triples"]])
    elif e.get("triples") and isinstance(e["triples"][0], dict) and "classification" in e["triples"][0]:
        support = [t for t in e["triples"] if t.get("classification") == "support"]
        nodes, rels, graph = triples_from_list(support)
    else:
        nodes, rels, graph = triples_from_list(e.get("triples", []))

    rec["sample0"]["nodes"] = nodes
    rec["sample0"]["relations"] = rels
    rec["sample0"]["graph"] = graph

    # Add samples if available
    for i, blk in enumerate(e.get("sample_triples", []), start=1):
        sn, sr, sg = triples_from_block(blk)
        rec[f"sample{i}"] = {"nodes": sn, "relations": sr, "graph": sg}

    out.append(rec)

json.dump(out, open(save_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[OK] wrote {len(out)} entries -> {save_path}")
