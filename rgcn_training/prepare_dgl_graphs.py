# sambuaneesh-gca-1/rgcn_training/prepare_dgl_graphs.py
import json
import os

import dgl
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Configuration ---
INPUT_JSON_PATH = "/home/stealthspectre/iiith/GCA/Extract triples/processed/graphs_wiki.json"
OUTPUT_DGL_DIR = "/home/stealthspectre/iiith/GCA/rgcn_training/data/dgl_graphs_wiki"
OUTPUT_REL_MAP_PATH = "/home/stealthspectre/iiith/GCA/rgcn_training/data/relation2id_wiki.json"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print(f"Loading data from {INPUT_JSON_PATH}...")
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- 1. Build Relation Map ---
    print("Building relation map...")
    all_relations = set()
    for entry in data:
        # We collect relations from all samples, not just sample0
        for key, sample in entry.items():
            if isinstance(sample, dict) and "relations" in sample:
                for rel in sample["relations"]:
                    all_relations.add(rel)

    relation_list = sorted(list(all_relations))
    relation2id = {rel: i for i, rel in enumerate(relation_list)}
    relation2id["__UNK__"] = len(relation2id)  # Add an UNK token

    print(f"Found {len(relation2id)} unique relations.")
    os.makedirs(os.path.dirname(OUTPUT_REL_MAP_PATH), exist_ok=True)
    with open(OUTPUT_REL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(relation2id, f, indent=2)
    print(f"Saved relation map to {OUTPUT_REL_MAP_PATH}")

    # --- 2. Initialize Sentence Transformer ---
    print(f"Loading S-BERT model: {SBERT_MODEL}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(SBERT_MODEL, device=device)

    # --- 3. Create and Save DGL Graphs ---
    print(f"Processing {len(data)} entries and saving DGL graphs to {OUTPUT_DGL_DIR}...")
    os.makedirs(OUTPUT_DGL_DIR, exist_ok=True)

    for i, entry in enumerate(tqdm(data, desc="Creating DGL graphs")):
        # We only process 'sample0' as this is the primary graph for training
        sample0 = entry.get("sample0")
        if not sample0 or not sample0.get("nodes") or not sample0.get("graph"):
            print(f"Skipping entry {i} due to missing data in sample0.")
            continue

        nodes = sample0["nodes"]
        triples = sample0["graph"]

        # Create a mapping from node name to its index
        node2idx = {name: j for j, name in enumerate(nodes)}

        # Encode node features
        with torch.no_grad():
            node_feats = embedder.encode(nodes, convert_to_tensor=True, normalize_embeddings=True, device=device).cpu()

        # Prepare graph edges and relation types
        src_idx, dst_idx, rel_ids = [], [], []
        unk_id = relation2id["__UNK__"]
        for h, r, t in triples:
            if h in node2idx and t in node2idx:
                src_idx.append(node2idx[h])
                dst_idx.append(node2idx[t])
                rel_ids.append(relation2id.get(r, unk_id))

        if not src_idx:  # Skip if no valid edges found
            continue

        # Create DGL graph
        g = dgl.graph((torch.tensor(src_idx), torch.tensor(dst_idx)), num_nodes=len(nodes))
        g.ndata["feat"] = node_feats
        g.edata[dgl.ETYPE] = torch.tensor(rel_ids, dtype=torch.long)

        # Save the graph
        graph_filename = os.path.join(OUTPUT_DGL_DIR, f"graph_{i:04d}.dgl")
        dgl.save_graphs(graph_filename, [g])

    print("Finished preparing DGL graphs.")


if __name__ == "__main__":
    main()
