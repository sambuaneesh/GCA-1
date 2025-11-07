# Temporal Graph Network: Hallucination Detection in Multi-Turn Conversation

## Summary

### TGN Pipeline (Reproducible Steps)

DiaHalu can be replaced with any given dataset.

| Details | Script Name | Input File | Output File |
|---------|-------------|------------|-------------|
| Standardizes DiaHalu dataset from Excel to JSON format. | `Extract triples/standardize_diahalu.py` | `Extract triples/dataset/DiaHalu_V2.xlsx` | `Extract triples/dataset/DiaHalu_V2.json` |
| Extracts entities for each dialogue turn. | `Temporal Graph/extract_entities_per_dialogue.py` | `Extract triples/dataset/DiaHalu_V2.json` | `Temporal Graph/processed/diahalu_temporal.json` |
| Builds temporal graphs with entity and temporal edges. | `Temporal Graph/build_temporal_graphs.py` | `Temporal Graph/processed/diahalu_temporal.json` | `Temporal Graph/processed/dgl_temporal/` |
| Trains the Temporal Graph Network model. | `Temporal Graph/train_temporal.py` | `Temporal Graph/processed/dgl_temporal/` | `Temporal Graph/processed/temporal_entity_ckpt.pt` |
| Identifies misclassified samples in validation set. | `Temporal Graph/find_errors.py` | `Temporal Graph/processed/dgl_temporal/`, `Temporal Graph/processed/temporal_entity_ckpt.pt` | - |
| Checks attention weights for individual validation samples. | `Temporal Graph/infer_temporal.py` | `Temporal Graph/processed/dgl_temporal/`, `Temporal Graph/processed/temporal_entity_ckpt.pt`, `Temporal Graph/processed/diahalu_temporal.json` | - |
| Generates confusion matrix visualization for validation set. | `Temporal Graph/plot_confusion_matrix.py` | `Temporal Graph/processed/dgl_temporal/`, `Temporal Graph/processed/temporal_entity_ckpt.pt` | `Temporal Graph/processed/confusion_matrix.png` |

### GCA Pipeline (Reproducible Steps)

WikiBio can be replaced with any given dataset.

| Details | Script Name | Input File | Output File |
|---------|-------------|------------|-------------|
| Extracts triples from main text. | `GCA/Extract triples/process_dataset.py` | `GCA/Extract triples/dataset/WikiBio_dataset/wikibio.json` | `GCA/Extract triples/processed/out_triplets_wiki.json` |
| Extracts triples from samples. | `Reverse Verification of Triples/compare triples/sample.py` | `GCA/Extract triples/processed/out_triplets_wiki.json` | `GCA/Extract triples/processed/out_samples_wiki.json` |
| Verifies triplets of main text based on entailment from triplets of samples. | `Reverse Verification of Triples/compare triples/gpt4_compare.py` | `GCA/Extract triples/processed/out_samples_wiki.json` | `GCA/Extract triples/processed/out_supports_wiki_rr.json` |
| Marks triplets of main text as facts based on entailment. | `Reverse Verification of Triples/compare triples/fact_triples.py` | `GCA/Extract triples/processed/out_supports_wiki_rr.json` | Same file (in-place modification) |
| Reverse verification of all the triplets of main text. | `Reverse Verification of Triples/mask_relationship.py` | `GCA/Extract triples/processed/out_supports_wiki.json` | `GCA/Extract triples/processed/out_mask_wiki_rr.json` |
| Builds the graph using the triplets. | `Graph-based Contextual Consistency Comparison/extract_nodes&edges.py` | `GCA/Extract triples/processed/out_triplets_wiki.json` | `GCA/Extract triples/processed/graphs_wiki.json` |
| Generates relation mapping and DGL files. | `rgcn_training/prepare_dgl_graphs.py` | `GCA/Extract triples/processed/graphs_wiki.json` | `data/relation2id_wiki.json`, `data/dgl_graphs_wiki/` |
| Trains the RGCN for the Wikibio dataset. | `rgcn_training/train_gca_rgcn.py` | `GCA/rgcn_training/data/dgl_graphs_wiki/` | `GCA/checkpoints/gca_rgcn_wiki.ckpt` |
| Calculates the final score for each triplet of main text. | `Graph-based Contextual Consistency Comparison/score_with_rgcn.py` | `GCA/Extract triples/processed/graphs_wiki.json`, `data/relation2id_wiki.json`, `GCA/checkpoints/gca_rgcn_wiki.ckpt` | `GCA/Extract triples/processed/scored_wiki.json` |
| Calculates the classification metrics for different thresholds on the final score. | `wikiBio/cal_metric_gca.py` | `GCA/Extract triples/processed/scored_wiki.json` | - |

## How to Run

To execute any script from the pipelines above:

1. **Synchronize dependencies:**
    ```bash
    uv sync
    ```

2. **View script arguments:**
    ```bash
    uv run <script_name> --help
    ```

3. **Match the arguments** with the corresponding pipeline table (TGN or GCA) to identify required input files and expected outputs.

## Model Files

Pre-trained model files are available at: [Google Drive](https://drive.google.com/drive/folders/1vEhI2iKUKz7k-AE7fa7QHO9ncYXBd19g?usp=sharing).