import numpy as np
from src.kinematics import VehicleState, step
from src.l1_guidance import L1Guidance
from src.rule_based_nav import RuleBasedWaypointNav
from src.follower_controller import FollowerController
from src.formation import LeaderTrail
from src.metrics import cross_track_error


def build_leader_controller(method, waypoints, segments, L1=3.0, V=3.0):
    if method == "l1":
        return L1Guidance(segments, L1=L1, V=V, max_psi_dot=np.deg2rad(60))
    elif method == "rule_based":
        return RuleBasedWaypointNav(waypoints, capture_radius=1.0,
                                     kp_heading=1.5, max_psi_dot=np.deg2rad(60),
                                     use_speed_tiers=False, constant_V=V)
    else:
        raise ValueError("method harus 'l1' atau 'rule_based'")


def run_formation_simulation(method, waypoints, segments,
                              d0=5.0, leader_V=3.0,
                              leader_initial=None, follower_initial=None,
                              dt=0.05, t_max=60.0, capture_radius_finish=0.5):

    if leader_initial is None:
        leader_initial = VehicleState(0.0, 0.0, 0.0)
    if follower_initial is None:
        # Follower mulai di belakang leader sejauh d0, sedikit offset lateral
        follower_initial = VehicleState(-d0, 0.5, 0.0)

    leader_state = leader_initial
    follower_state = follower_initial

    leader_controller = build_leader_controller(method, waypoints, segments, V=leader_V)
    follower_controller = FollowerController(method=method, d0=d0, L1=3.0,
                                              kp_heading=1.5,
                                              max_psi_dot=np.deg2rad(60))

    leader_trail = LeaderTrail()
    leader_trail.append(leader_state.position())

    n_steps = int(t_max / dt)

    time_hist = np.zeros(n_steps)
    leader_x = np.zeros(n_steps)
    leader_y = np.zeros(n_steps)
    follower_x = np.zeros(n_steps)
    follower_y = np.zeros(n_steps)
    formation_dist = np.zeros(n_steps)
    leader_cte = np.zeros(n_steps)

    finished = False
    last_step = n_steps

    for k in range(n_steps):
        psi_dot_L, V_L, _ = leader_controller.compute_control(leader_state)
        psi_dot_F, V_F, _target, dist_LF = follower_controller.compute_control(
            follower_state, leader_trail, leader_state.position())

        time_hist[k] = k * dt
        leader_x[k] = leader_state.x
        leader_y[k] = leader_state.y
        follower_x[k] = follower_state.x
        follower_y[k] = follower_state.y
        formation_dist[k] = dist_LF
        leader_cte[k] = cross_track_error(leader_state.position(), segments)

        if leader_controller.is_finished(leader_state, capture_radius=capture_radius_finish):
            finished = True
            last_step = k + 1
            break

        leader_state = step(leader_state, V_L, psi_dot_L, dt, max_psi_dot=None)
        follower_state = step(follower_state, V_F, psi_dot_F, dt, max_psi_dot=None)
        leader_trail.append(leader_state.position())

    return {
        "time": time_hist[:last_step],
        "leader_x": leader_x[:last_step],
        "leader_y": leader_y[:last_step],
        "follower_x": follower_x[:last_step],
        "follower_y": follower_y[:last_step],
        "formation_dist": formation_dist[:last_step],
        "leader_cte": leader_cte[:last_step],
        "finished": finished,
    }
