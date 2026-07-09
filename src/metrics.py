import numpy as np
from src.l1_guidance import closest_point_on_segment


def cross_track_error(pos, segments, active_seg_idx_hint=0):
 
    min_dist = np.inf
    for (a, b) in segments:
        proj, _ = closest_point_on_segment(pos, a, b)
        d = np.linalg.norm(pos - proj)
        if d < min_dist:
            min_dist = d
    return min_dist


def compute_metrics(cte_history, corridor_width=1.5, steady_state_frac=0.3):
 
    cte = np.asarray(cte_history)
    n = len(cte)
    n_tail = max(int(n * steady_state_frac), 1)

    mean_cte = float(np.mean(cte))
    rmse_cte = float(np.sqrt(np.mean(cte ** 2)))
    max_cte = float(np.max(cte))
    steady_state_error = float(np.mean(cte[-n_tail:]))
    accuracy_pct = float(np.mean(cte <= corridor_width) * 100.0)

    return {
        "mean_cte": mean_cte,
        "rmse_cte": rmse_cte,
        "max_cte": max_cte,
        "steady_state_error": steady_state_error,
        "accuracy_pct": accuracy_pct,
    }


def compute_formation_metrics(formation_dist_history, d0=5.0, tolerance=1.0):
 
    dist = np.asarray(formation_dist_history)
    error = np.abs(dist - d0)

    return {
        "mean_error": float(np.mean(error)),
        "rmse_error": float(np.sqrt(np.mean(error ** 2))),
        "max_error": float(np.max(error)),
        "within_tolerance_pct": float(np.mean(error <= tolerance) * 100.0),
    }
