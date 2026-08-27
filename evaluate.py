#!/usr/bin/env python3
"""
evaluate.py — Industry-Grade Road Segmentation Evaluation Pipeline
====================================================================
Comprehensive evaluation of the NetSight road segmentation model with:
- Pixel-level metrics: IoU, Dice, Relaxed IoU, Precision, Recall, F1
- Graph-level metrics: Topological path error, connectivity improvement
- Optional TTA (Test-Time Augmentation) for fair comparison
- CSV export for tracking experiments

Author: NetSight Project (ISRO NNRMS / Disaster Management Framework)
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import networkx as nx
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from train import RoadSegDataset, get_val_transform
from my_agents.src.model import ResNetUNet, OcclusionRobustModel, RoadSegModel, TestTimeAugmentor
from my_agents.src.mask_to_graph import mask_to_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("netsight.evaluate")


def compute_relaxed_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, buffer_px: int = 3) -> float:
    """
    Computes Length-Complete / Relaxed IoU.
    Introduces a tolerance buffer. If the predicted road pixel falls within the buffer zone 
    of the ground truth road, it counts as a true positive.
    """
    if gt_mask.sum() == 0 and pred_mask.sum() == 0:
        return 1.0
    if gt_mask.sum() == 0 or pred_mask.sum() == 0:
        return 0.0

    # Distance from background to nearest road pixel
    dist_gt = distance_transform_edt(1 - gt_mask)
    dist_pred = distance_transform_edt(1 - pred_mask)

    # Relaxed masks
    relaxed_gt = (dist_gt <= buffer_px).astype(bool)
    relaxed_pred = (dist_pred <= buffer_px).astype(bool)

    tp_pred = np.logical_and(pred_mask, relaxed_gt).sum()
    tp_gt = np.logical_and(gt_mask, relaxed_pred).sum()
    
    precision = tp_pred / (pred_mask.sum() + 1e-7)
    recall = tp_gt / (gt_mask.sum() + 1e-7)
    
    iou = (precision * recall) / (precision + recall - precision * recall + 1e-7)
    return float(iou)


def compute_pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Compute comprehensive pixel-level metrics.

    Returns dict with: iou, dice, precision, recall, f1, relaxed_iou
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    intersection = tp
    union = tp + fp + fn

    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / (2 * tp + fp + fn + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    f1 = (2 * precision * recall) / (precision + recall + 1e-7)

    relaxed_iou = compute_relaxed_iou(
        pred_mask.astype(np.uint8), gt_mask.astype(np.uint8), buffer_px=4,
    )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "relaxed_iou": float(relaxed_iou),
    }


def get_largest_cc_size(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    return len(components[0])


def compute_topological_accuracy(G_gt: nx.Graph, G_pred: nx.Graph, num_samples: int = 10) -> float:
    """
    Samples random point pairs on the ground-truth OSM graph and compares their shortest path
    lengths to the predicted graph.
    """
    if G_gt.number_of_nodes() < 2 or G_pred.number_of_nodes() < 2:
        return 1.0 # Max error

    # Find the largest connected component of GT to ensure path exists
    components = sorted(nx.connected_components(G_gt), key=len, reverse=True)
    G_gt_cc = G_gt.subgraph(components[0])
    
    nodes_gt = list(G_gt_cc.nodes(data=True))
    nodes_pred = list(G_pred.nodes(data=True))
    
    if len(nodes_gt) < 2:
        return 1.0
        
    errors = []
    
    for _ in range(num_samples):
        # Sample two random nodes from GT
        n1, n2 = random.sample(nodes_gt, 2)
        pos1, pos2 = np.array(n1[1]['pos']), np.array(n2[1]['pos'])
        
        # Shortest path on GT
        try:
            path_len_gt = nx.shortest_path_length(G_gt_cc, source=n1[0], target=n2[0], weight='w')
        except nx.NetworkXNoPath:
            continue
            
        if path_len_gt < 1e-5:
            continue
            
        # Find nearest nodes in Pred graph
        def nearest_node(pos, nodes_list):
            best_nid, best_dist = None, float('inf')
            for nid, data in nodes_list:
                d = np.linalg.norm(np.array(data['pos']) - pos)
                if d < best_dist:
                    best_dist = d
                    best_nid = nid
            return best_nid, best_dist
            
        pred_n1, dist1 = nearest_node(pos1, nodes_pred)
        pred_n2, dist2 = nearest_node(pos2, nodes_pred)
        
        if dist1 > 50 or dist2 > 50: # If prediction is totally missing this area
            errors.append(1.0)
            continue
            
        try:
            path_len_pred = nx.shortest_path_length(G_pred, source=pred_n1, target=pred_n2, weight='w')
            err = abs(path_len_pred - path_len_gt) / path_len_gt
            errors.append(min(err, 1.0))
        except nx.NetworkXNoPath:
            errors.append(1.0) # Penalty for disconnected path in pred
            
    if not errors:
        return 0.0
    return sum(errors) / len(errors)


def evaluate(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Running evaluation on {device}")
    
    dataset = RoadSegDataset(args.data_dir, transform=get_val_transform(512))
    log.info(f"Loaded dataset with {len(dataset)} images from {args.data_dir}")
    
    # Model loading
    if args.model_type == "industry":
        model = RoadSegModel(
            encoder_name=args.encoder_name,
            encoder_weights=None, in_channels=3, classes=1,
        ).to(device)
    elif args.model_type == "resnet":
        model = ResNetUNet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
    else:
        model = OcclusionRobustModel(architecture=args.model_type, encoder_name="mit_b3", encoder_weights=None, in_channels=3, classes=1).to(device)
        
    try:
        model.load_state_dict(torch.load(args.model_weights, map_location=device))
        log.info(f"Loaded weights from {args.model_weights}")
    except Exception as e:
        log.error(f"Failed to load weights: {e}")
        log.info("Trying with strict=False...")
        try:
            model.load_state_dict(torch.load(args.model_weights, map_location=device), strict=False)
            log.info("Loaded with strict=False (some layers may be missing)")
        except Exception as e2:
            log.error(f"Still failed: {e2}")
            return
        
    model.eval()

    # Optional TTA
    tta = None
    if args.use_tta:
        tta = TestTimeAugmentor(model, device=str(device))
        log.info("TTA enabled (4x inference per sample)")
    
    metrics = {
        "iou": [],
        "dice": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "relaxed_iou": [],
        "conn_ratio": [],
        "topo_error": [],
    }
    
    # Sample subset for topological evaluation
    eval_indices = list(range(len(dataset)))
    if len(eval_indices) > args.max_samples:
        random.seed(42)
        eval_indices = random.sample(eval_indices, args.max_samples)
        log.info(f"Randomly subsampled {args.max_samples} images for evaluation")

    per_sample_results = []

    with torch.no_grad():
        for idx in tqdm(eval_indices, desc="Evaluating Pipeline"):
            img_tensor, gt_tensor = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            gt_mask = gt_tensor.squeeze().cpu().numpy()
            
            # 1. Inference
            if tta is not None:
                probs = tta.predict(img_tensor).squeeze().cpu().numpy()
            else:
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(img_tensor)
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            pred_mask = (probs > 0.5).astype(np.uint8)
            gt_mask_uint8 = (gt_mask > 0.5).astype(np.uint8)
            
            # 2. Pixel Metrics
            px_metrics = compute_pixel_metrics(pred_mask, gt_mask_uint8)
            
            for key in ["iou", "dice", "precision", "recall", "f1", "relaxed_iou"]:
                metrics[key].append(px_metrics[key])
            
            # 3. Graph Metrics
            G_gt, _, _ = mask_to_graph(gt_mask_uint8, heal=False, prune_min_length=0)
            G_pred_raw, _, _ = mask_to_graph(pred_mask, heal=False, prune_min_length=5.0)
            cc_pre = get_largest_cc_size(G_pred_raw)
            
            G_pred_healed, _, _ = mask_to_graph(pred_mask, heal=True, prune_min_length=5.0)
            cc_post = get_largest_cc_size(G_pred_healed)
            
            if cc_pre > 0:
                conn_ratio = (cc_post - cc_pre) / cc_pre * 100.0
            else:
                conn_ratio = 0.0
                
            metrics["conn_ratio"].append(conn_ratio)
            
            topo_err = compute_topological_accuracy(G_gt, G_pred_healed, num_samples=5)
            metrics["topo_error"].append(topo_err)

            # Per-sample record for CSV
            per_sample_results.append({
                "index": idx,
                **px_metrics,
                "conn_ratio": conn_ratio,
                "topo_error": topo_err,
            })

    # 4. Generate Final Report
    print("\n" + "=" * 70)
    print(" " * 10 + "ISRO NNRMS - COMPREHENSIVE EVALUATION REPORT")
    print("=" * 70)
    print(f"  Model Architecture : {args.model_type}")
    if args.model_type == "industry":
        print(f"  Encoder            : {args.encoder_name}")
    print(f"  Weights File       : {args.model_weights}")
    print(f"  Dataset            : {args.data_dir}")
    print(f"  Samples Evaluated  : {len(eval_indices)}")
    print(f"  TTA Enabled        : {'Yes (4x)' if args.use_tta else 'No'}")
    print("-" * 70)
    print("  PIXEL-LEVEL METRICS:")
    print(f"    1. Mean IoU (Strict)         : {np.mean(metrics['iou']):.4f}  ±{np.std(metrics['iou']):.4f}")
    print(f"    2. Mean Dice Score           : {np.mean(metrics['dice']):.4f}  ±{np.std(metrics['dice']):.4f}")
    print(f"    3. Mean Precision            : {np.mean(metrics['precision']):.4f}")
    print(f"    4. Mean Recall               : {np.mean(metrics['recall']):.4f}")
    print(f"    5. Mean F1 Score             : {np.mean(metrics['f1']):.4f}")
    print(f"    6. Relaxed IoU (4px Buffer)  : {np.mean(metrics['relaxed_iou']):.4f}  <-- Occlusion/Alignment Resilience")
    print("-" * 70)
    print("  GRAPH-LEVEL METRICS:")
    print(f"    7. Connectivity Boost        : +{np.mean(metrics['conn_ratio']):.2f}%   <-- MST Healing Efficacy")
    print(f"    8. Topological Path Error    : {np.mean(metrics['topo_error'])*100:.2f}%    <-- Graph Routability vs OSM")
    print("=" * 70)

    # 5. Export to CSV
    if args.csv_output:
        csv_path = Path(args.csv_output)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_sample_results[0].keys())
            writer.writeheader()
            writer.writerows(per_sample_results)

        # Also write summary row
        summary_path = csv_path.parent / f"{csv_path.stem}_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "mean", "std", "min", "max"])
            for key in ["iou", "dice", "precision", "recall", "f1", "relaxed_iou", "conn_ratio", "topo_error"]:
                vals = metrics[key]
                writer.writerow([key, f"{np.mean(vals):.6f}", f"{np.std(vals):.6f}",
                                 f"{np.min(vals):.6f}", f"{np.max(vals):.6f}"])

        log.info(f"Results exported to {csv_path} and {summary_path}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NetSight Evaluation Pipeline")
    parser.add_argument("--data_dir", type=str, default="training_data")
    parser.add_argument("--model_weights", type=str, default="my_agents/weights/model_weights.pth")
    parser.add_argument("--model_type", type=str, default="industry",
                        choices=["industry", "resnet", "unetplusplus", "manet"])
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b4",
                        help="Encoder for 'industry' model type")
    parser.add_argument("--use_tta", action="store_true", default=False,
                        help="Enable Test-Time Augmentation during evaluation")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum samples for graph evaluation")
    parser.add_argument("--csv_output", type=str, default="evaluation_results.csv",
                        help="Path to export per-sample CSV results")
    args = parser.parse_args()
    evaluate(args)
