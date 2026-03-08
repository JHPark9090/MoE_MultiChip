"""
Test whether Limbic_TempPole and Limbic_OFC subtypes are robust to K choice.

For each v5 heterogeneity analysis, loads the subject_representations.npz,
sweeps K=2..6 at Level 3 (ROI), and checks whether any cluster has
Limbic_TempPole or Limbic_OFC as its dominant +ADHD network.

This directly tests whether our K=3 findings are artifacts of K choice.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
sys.path.insert(0, '.')
from models.yeo17_networks import YEO17_HCP180
from dataloaders.hcp_mmp1_labels import get_roi_info, get_roi_name


def check_limbic_subtypes(signed_roi_scores, cluster_ids, n_clusters):
    """
    For each cluster, compute mean signed ROI score per network.
    Check if any cluster has Limbic_TempPole or Limbic_OFC as top +ADHD network.
    Also return top 5 +ADHD ROIs per cluster.
    """
    results = []
    for c in range(n_clusters):
        mask = cluster_ids == c
        n_subj = mask.sum()
        mean_signed = signed_roi_scores[mask].mean(axis=0)

        # Network-level signed scores
        net_scores = {}
        for net_name, roi_indices in YEO17_HCP180.items():
            net_scores[net_name] = float(mean_signed[roi_indices].mean())

        # Top +ADHD network
        top_pos_net = max(net_scores, key=net_scores.get)
        top_pos_score = net_scores[top_pos_net]

        # Top 5 +ADHD ROIs
        top5_idx = np.argsort(mean_signed)[-5:][::-1]
        top5_rois = []
        # Build ROI-to-network map
        roi_to_net = {}
        for net_name, indices in YEO17_HCP180.items():
            for idx in indices:
                roi_to_net[idx] = net_name
        for idx in top5_idx:
            roi_name = get_roi_name(idx)
            top5_rois.append({
                'roi_idx': int(idx),
                'region': roi_name,
                'network': roi_to_net.get(idx, 'Unknown'),
                'signed_score': float(mean_signed[idx])
            })

        results.append({
            'cluster': c,
            'n_subjects': int(n_subj),
            'pct': float(100 * n_subj / len(cluster_ids)),
            'top_pos_network': top_pos_net,
            'top_pos_score': top_pos_score,
            'has_limbic_temppole': top_pos_net == 'Limbic_TempPole',
            'has_limbic_ofc': top_pos_net == 'Limbic_OFC',
            'top5_rois': top5_rois,
            'limbic_temppole_score': net_scores.get('Limbic_TempPole', 0),
            'limbic_ofc_score': net_scores.get('Limbic_OFC', 0),
        })

    return results


def main():
    dirs = [
        ('Classical 4-expert', 'heterogeneity_v5_classical_adhd_3'),
        ('Classical 2-expert', 'heterogeneity_v5_classical_adhd_2'),
        ('Quantum 4-expert', 'heterogeneity_v5_quantum_8q_d3_adhd_3'),
        ('Quantum 2-expert', 'heterogeneity_v5_quantum_8q_d3_adhd_2'),
    ]

    k_range = range(2, 7)  # K=2,3,4,5,6

    print("=" * 80)
    print("SUBTYPE STABILITY TEST: Do Limbic subtypes persist across K values?")
    print("=" * 80)

    # Summary tracking
    temppole_found = {k: [] for k in k_range}
    ofc_found = {k: [] for k in k_range}

    for model_name, dir_name in dirs:
        base = Path(f"analysis/{dir_name}")
        npz = np.load(base / "subject_representations.npz", allow_pickle=True)

        labels = npz['labels']
        adhd_mask = labels == 1
        signed_roi = npz['signed_roi_scores'][adhd_mask]
        n_adhd = adhd_mask.sum()

        print(f"\n{'─' * 80}")
        print(f"  {model_name} (N_ADHD+={n_adhd})")
        print(f"{'─' * 80}")

        # Silhouette sweep at ROI level
        print(f"\n  ROI-Level Silhouette Sweep:")
        roi_scores_abs = npz['roi_scores'][adhd_mask]
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=2025)
            cids = km.fit_predict(roi_scores_abs)
            sil = silhouette_score(roi_scores_abs, cids)
            print(f"    K={k}: silhouette={sil:.4f}")

        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=2025)
            cluster_ids = km.fit_predict(roi_scores_abs)

            cluster_results = check_limbic_subtypes(signed_roi, cluster_ids, k)

            has_tp = any(c['has_limbic_temppole'] for c in cluster_results)
            has_ofc = any(c['has_limbic_ofc'] for c in cluster_results)

            if has_tp:
                temppole_found[k].append(model_name)
            if has_ofc:
                ofc_found[k].append(model_name)

            # Print details
            print(f"\n  K={k}:")
            for c in cluster_results:
                marker = ''
                if c['has_limbic_temppole']:
                    marker = ' *** LIMBIC_TEMPPOLE ***'
                elif c['has_limbic_ofc']:
                    marker = ' *** LIMBIC_OFC ***'

                top5_str = ', '.join(
                    f"{r['region']}({r['network']})" for r in c['top5_rois']
                )
                print(f"    Cluster {c['cluster']}: N={c['n_subjects']} "
                      f"({c['pct']:.1f}%) | Top +ADHD net: {c['top_pos_network']} "
                      f"({c['top_pos_score']:+.6f}){marker}")
                print(f"      Top 5 +ADHD ROIs: {top5_str}")
                print(f"      Limbic_TempPole={c['limbic_temppole_score']:+.6f}, "
                      f"Limbic_OFC={c['limbic_ofc_score']:+.6f}")

    # ───────────────────────────────────────────────────────────
    # Summary
    # ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("STABILITY SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nLimbic_TempPole subtype found (top +ADHD network in any cluster):")
    for k in k_range:
        models = temppole_found[k]
        status = f"{len(models)}/4 models" if models else "NOT FOUND"
        names = ', '.join(models) if models else ''
        print(f"  K={k}: {status}  {names}")

    print(f"\nLimbic_OFC subtype found (top +ADHD network in any cluster):")
    for k in k_range:
        models = ofc_found[k]
        status = f"{len(models)}/4 models" if models else "NOT FOUND"
        names = ', '.join(models) if models else ''
        print(f"  K={k}: {status}  {names}")

    # Overall robustness
    tp_robust = all(len(temppole_found[k]) >= 3 for k in k_range)
    ofc_robust = all(len(ofc_found[k]) >= 3 for k in k_range)

    print(f"\n{'─' * 80}")
    print(f"Limbic_TempPole robust across all K values? {'YES' if tp_robust else 'NO'}")
    print(f"Limbic_OFC robust across all K values?      {'YES' if ofc_robust else 'NO'}")
    print(f"{'─' * 80}")


if __name__ == '__main__':
    main()
