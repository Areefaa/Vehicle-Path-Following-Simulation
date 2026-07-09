import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.trajectory import generate_l_shape_waypoints, path_segments, dense_path_points
from src.simulate_formation import run_formation_simulation
from src.metrics import compute_formation_metrics, compute_metrics
from src.kinematics import VehicleState

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ---- Parameter global simulasi ----
S1, S2 = 15.0, 15.0
D0 = 5.0          # jarak formasi nominal (m)
DT = 0.05         # langkah waktu simulasi (s)
T_MAX = 40.0      # durasi maksimum simulasi (s)
CORRIDOR_WIDTH = 1.5   # lebar koridor toleransi CTE leader (m)
FORMATION_TOLERANCE = 1.0  # toleransi jarak formasi (m)

METHOD_LABELS = {
    "l1": "L1 Guidance",
    "rule_based": "Rule-Based Waypoint Nav",
}


def run_scenario(method, waypoints, segments):
    """Menjalankan simulasi formasi untuk satu metode navigasi."""
    result = run_formation_simulation(
        method, waypoints, segments, d0=D0, leader_V=3.0,
        leader_initial=VehicleState(0.0, 0.0, 0.0),
        follower_initial=VehicleState(-D0, 0.5, 0.0),
        dt=DT, t_max=T_MAX)

    formation_metrics = compute_formation_metrics(
        result["formation_dist"], d0=D0, tolerance=FORMATION_TOLERANCE)
    cte_metrics = compute_metrics(result["leader_cte"], corridor_width=CORRIDOR_WIDTH)

    return result, formation_metrics, cte_metrics


def print_summary(method, result, formation_metrics, cte_metrics):
    label = METHOD_LABELS[method]
    print(f"\n[{label}] (leader & follower)")
    print(f"  Selesai                                : {result['finished']}")
    print(f"  Waktu tempuh                            : {result['time'][-1]:.2f} s")
    print(f"  Leader  - Mean CTE                      : {cte_metrics['mean_cte']:.3f} m")
    print(f"  Leader  - Akurasi (CTE dlm koridor)      : {cte_metrics['accuracy_pct']:.2f} %")
    print(f"  Formasi - Mean |error jarak ke d0|        : {formation_metrics['mean_error']:.3f} m")
    print(f"  Formasi - RMSE error jarak                : {formation_metrics['rmse_error']:.3f} m")
    print(f"  Formasi - Max error jarak                 : {formation_metrics['max_error']:.3f} m")
    print(f"  Formasi - % waktu dlm toleransi (+-{FORMATION_TOLERANCE} m) : {formation_metrics['within_tolerance_pct']:.2f} %")


def plot_static_path(method, result, waypoints, ref_path):
    label = METHOD_LABELS[method]
    plt.figure(figsize=(7, 7))
    plt.plot(ref_path[:, 0], ref_path[:, 1], "k--", linewidth=1.5, label="Lintasan referensi (L-shape)")
    plt.plot(result["leader_x"], result["leader_y"], color="tab:blue", linewidth=2, label="Leader")
    plt.plot(result["follower_x"], result["follower_y"], color="tab:green", linewidth=2, label="Follower")
    plt.scatter(*zip(*waypoints), color="red", zorder=5, label="Waypoint")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Formasi Leader-Follower - {label}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"formation_path_{method}.png"), dpi=150)
    plt.close()


def plot_formation_distance_comparison(result_l1, result_rb):
    plt.figure(figsize=(9, 5))
    plt.plot(result_l1["time"], result_l1["formation_dist"], label=METHOD_LABELS["l1"], linewidth=2)
    plt.plot(result_rb["time"], result_rb["formation_dist"], label=METHOD_LABELS["rule_based"], linewidth=2)
    plt.axhline(D0, color="gray", linestyle=":", label=f"Jarak formasi nominal d0 = {D0} m")
    plt.xlabel("Waktu (s)")
    plt.ylabel("Jarak Follower-ke-Leader (m)")
    plt.title("Perbandingan Jarak Formasi terhadap Waktu")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "formation_distance.png"), dpi=150)
    plt.close()


def animate_formation(method, result, waypoints, ref_path, step_skip=4, fps=30):
    label = METHOD_LABELS[method]

    leader_x, leader_y = result["leader_x"], result["leader_y"]
    follower_x, follower_y = result["follower_x"], result["follower_y"]
    n_frames = len(leader_x)
    frame_idx = np.arange(0, n_frames, step_skip)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(ref_path[:, 0], ref_path[:, 1], "k--", linewidth=1.5, label="Lintasan referensi (L-shape)")
    ax.scatter(*zip(*waypoints), color="red", zorder=5, label="Waypoint")

    leader_trail_line, = ax.plot([], [], color="tab:blue", linewidth=2, label="Leader")
    follower_trail_line, = ax.plot([], [], color="tab:green", linewidth=2, label="Follower")
    leader_point, = ax.plot([], [], "o", color="tab:blue", markersize=9)
    follower_point, = ax.plot([], [], "o", color="tab:green", markersize=9)
    link_line, = ax.plot([], [], color="gray", linewidth=1, linestyle="-", alpha=0.6)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Animasi Formasi Leader-Follower - {label}")
    ax.legend(loc="upper left")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    margin = 2.0
    all_x = np.concatenate([ref_path[:, 0], leader_x, follower_x])
    all_y = np.concatenate([ref_path[:, 1], leader_y, follower_y])
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    def init():
        leader_trail_line.set_data([], [])
        follower_trail_line.set_data([], [])
        leader_point.set_data([], [])
        follower_point.set_data([], [])
        link_line.set_data([], [])
        return leader_trail_line, follower_trail_line, leader_point, follower_point, link_line

    def update(frame):
        i = frame_idx[frame]
        leader_trail_line.set_data(leader_x[:i + 1], leader_y[:i + 1])
        follower_trail_line.set_data(follower_x[:i + 1], follower_y[:i + 1])
        leader_point.set_data([leader_x[i]], [leader_y[i]])
        follower_point.set_data([follower_x[i]], [follower_y[i]])
        link_line.set_data([leader_x[i], follower_x[i]], [leader_y[i], follower_y[i]])
        return leader_trail_line, follower_trail_line, leader_point, follower_point, link_line

    ani = animation.FuncAnimation(fig, update, frames=len(frame_idx),
                                   init_func=init, blit=True, interval=1000 / fps)

    out_path = os.path.join(RESULTS_DIR, f"formation_animation_{method}.gif")
    ani.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Animasi disimpan  : {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    waypoints = generate_l_shape_waypoints(s1=S1, s2=S2)
    segments = path_segments(waypoints)
    ref_path = dense_path_points(waypoints)

    print("=" * 65)
    print("SIMULASI FORMASI LEADER-FOLLOWER - LINTASAN L-SHAPE")
    print(f"S1 = {S1} m, S2 = {S2} m, jarak formasi nominal d0 = {D0} m")
    print("=" * 65)

    result_l1, fm_l1, cte_l1 = run_scenario("l1", waypoints, segments)
    print_summary("l1", result_l1, fm_l1, cte_l1)
    plot_static_path("l1", result_l1, waypoints, ref_path)
    animate_formation("l1", result_l1, waypoints, ref_path)

    result_rb, fm_rb, cte_rb = run_scenario("rule_based", waypoints, segments)
    print_summary("rule_based", result_rb, fm_rb, cte_rb)
    plot_static_path("rule_based", result_rb, waypoints, ref_path)
    animate_formation("rule_based", result_rb, waypoints, ref_path)

    plot_formation_distance_comparison(result_l1, result_rb)

    print("=" * 65)
    print(f"\nSemua grafik & animasi disimpan di folder: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
