uv run rgcn_training/train_gca_rgcn.py \
    --data-dir /home/stealthspectre/iiith/GCA/rgcn_training/data/dgl_graphs_wiki \
    --checkpoint-out /home/stealthspectre/iiith/GCA/checkpoints/gca_rgcn_wiki.ckpt \
    --self-loop

uv run rgcn_training/train_gca_rgcn.py \
    --data-dir /home/stealthspectre/iiith/GCA/rgcn_training/data/dgl_graphs_phd \
    --checkpoint-out /home/stealthspectre/iiith/GCA/checkpoints/gca_rgcn_phd.ckpt \
    --self-loop

uv run rgcn_training/train_gca_rgcn.py \
    --data-dir /home/stealthspectre/iiith/GCA/rgcn_training/data/dgl_graphs_diahalu \
    --checkpoint-out /home/stealthspectre/iiith/GCA/checkpoints/gca_rgcn_diahalu.ckpt \
    --self-loop


----------------------------------------------------------------------------------------------


uv run "Graph-based Contextual Consistency Comparison/score_with_rgcn.py" \
  --ckpt "/home/stealthspectre/iiith/GCA/checkpoints/gca_rgcn_wiki.ckpt" \
  --relation-map "/home/stealthspectre/iiith/GCA/rgcn_training/data/relation2id_wiki.json" \
  --input "/home/stealthspectre/iiith/GCA/Extract triples/processed/graphs_wiki.json" \
  --output "/home/stealthspectre/iiith/GCA/Extract triples/processed/scored_wiki.json"

uv run "Graph-based Contextual Consistency Comparison/score_with_rgcn.py" \
  --ckpt "/home/stealthspectre/iiith/GCA/checkpoints/gca_rgcn_phd.ckpt" \
  --relation-map "/home/stealthspectre/iiith/GCA/rgcn_training/data/relation2id_phd.json" \
  --input "/home/stealthspectre/iiith/GCA/Extract triples/processed/graphs_phd.json" \
  --output "/home/stealthspectre/iiith/GCA/Extract triples/processed/scored_phd.json"

uv run "Graph-based Contextual Consistency Comparison/score_with_rgcn.py" \
  --ckpt "/home/stealthspectre/iiith/GCA/checkpoints/gca_rgcn_diahalu.ckpt" \
  --relation-map "/home/stealthspectre/iiith/GCA/rgcn_training/data/relation2id_diahalu.json" \
  --input "/home/stealthspectre/iiith/GCA/Extract triples/processed/graphs_diahalu.json" \
  --output "/home/stealthspectre/iiith/GCA/Extract triples/processed/scored_diahalu.json"
