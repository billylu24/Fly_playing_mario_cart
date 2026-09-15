"""Basic consistency checks using columns discovered by inspect_malecns.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
data = root / "data" / "malecns"
a = pd.read_feather(data / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
n = pd.read_feather(data / "body-neurotransmitters-male-cns-v1.0.feather")
w = pd.read_feather(data / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
ids = np.union1d(w.body_pre.unique(), w.body_post.unique())
annotation_ids = a.bodyId.to_numpy()
nt_ids = n.body.to_numpy()
ann_hit = np.isin(ids, annotation_ids)
nt_hit = np.isin(ids, nt_ids)
annotated_graph_ids = ids[ann_hit]
pre_ann = w.body_pre.isin(annotation_ids).to_numpy()
post_ann = w.body_post.isin(annotation_ids).to_numpy()
values, counts = np.unique(w.weight.to_numpy(), return_counts=True)
cumulative = counts.cumsum()
quantiles = {str(q): int(values[np.searchsorted(cumulative, max(1, int(q * len(w))))])
             for q in [0, 0.5, 0.9, 0.99, 1]}
report = {
    "graph_edges": len(w), "graph_unique_body_segment_ids": len(ids),
    "source_ids": w.body_pre.nunique(), "target_ids": w.body_post.nunique(),
    "graph_id_min": int(ids.min()), "graph_id_max": int(ids.max()),
    "zero_or_negative_graph_ids": int((ids <= 0).sum()),
    "weight_quantiles_nearest_rank": quantiles,
    "nonpositive_weights": int((w.weight <= 0).sum()),
    "weight_one_edges": int((w.weight == 1).sum()),
    "weight_sum": int(w.weight.sum()),
    "self_loop_edges": int((w.body_pre == w.body_post).sum()),
    "annotation_rows": len(a), "annotation_unique_ids": a.bodyId.nunique(),
    "annotation_duplicate_ids": int(a.bodyId.duplicated().sum()),
    "annotation_ids_in_graph": int(np.isin(annotation_ids, ids).sum()),
    "graph_annotation_coverage_count": int(ann_hit.sum()),
    "graph_annotation_coverage_fraction": float(ann_hit.mean()),
    "edges_with_both_endpoints_annotated": int((pre_ann & post_ann).sum()),
    "nt_rows": len(n), "nt_unique_ids": n.body.nunique(),
    "nt_duplicate_ids": int(n.body.duplicated().sum()),
    "nt_ids_in_graph": int(np.isin(nt_ids, ids).sum()),
    "graph_nt_coverage_count": int(nt_hit.sum()),
    "graph_nt_coverage_fraction": float(nt_hit.mean()),
    "annotated_graph_nt_coverage_count": int(np.isin(annotated_graph_ids, nt_ids).sum()),
    "annotated_graph_nt_coverage_fraction": float(np.isin(annotated_graph_ids, nt_ids).mean()),
    "annotation_type_nonnull": int(a.type.notna().sum()),
    "annotation_superclass_counts": a.superclass.value_counts(dropna=False).to_dict(),
    "annotation_status_counts": a.status.value_counts(dropna=False).to_dict(),
    "annotation_statusLabel_counts": a.statusLabel.value_counts(dropna=False).to_dict(),
    "nt_predicted_values": n.predicted_nt.value_counts(dropna=False).to_dict(),
    "nt_consensus_values": n.consensus_nt.value_counts(dropna=False).to_dict(),
    "nt_confidence_min": float(n.predicted_nt_confidence.min()),
    "nt_confidence_max": float(n.predicted_nt_confidence.max()),
    "nt_confidence_null": int(n.predicted_nt_confidence.isna().sum()),
    "nt_confidence_outside_0_1": int(((n.predicted_nt_confidence < 0) | (n.predicted_nt_confidence > 1)).sum()),
}
text = json.dumps(report, indent=2, ensure_ascii=False, default=int)
print(text, flush=True)
(root / "tests" / "malecns_summary.json").write_text(text + "\n")
