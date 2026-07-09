import numpy as np
from src.kinematics import wrap_angle
from src.formation import formation_speed_rule


class FollowerController:
    def __init__(self, method="l1", d0=5.0, L1=3.0,
                 kp_heading=1.5, max_psi_dot=np.deg2rad(60)):
    
        assert method in ("l1", "rule_based")
        self.method = method
        self.d0 = d0
        self.L1 = L1
        self.kp_heading = kp_heading
        self.max_psi_dot = max_psi_dot

    def compute_control(self, follower_state, leader_trail, leader_pos):
       
        pos = follower_state.position()
        target = leader_trail.point_at_arclength_behind(self.d0)

        distance_to_leader = np.linalg.norm(leader_pos - pos)
        V = formation_speed_rule(distance_to_leader)

        vec = target - pos
        vec_dist = np.linalg.norm(vec)

        if vec_dist < 1e-6:
            return 0.0, V, target, distance_to_leader

        if self.method == "l1":
            l1_angle = np.arctan2(vec[1], vec[0])
            eta = wrap_angle(l1_angle - follower_state.psi)
            a_s = (2 * V ** 2 / self.L1) * np.sin(eta)
            psi_dot = a_s / V if V > 1e-6 else 0.0
        else:  # rule_based
            bearing = np.arctan2(vec[1], vec[0])
            heading_error = wrap_angle(bearing - follower_state.psi)
            psi_dot = self.kp_heading * heading_error

        psi_dot = np.clip(psi_dot, -self.max_psi_dot, self.max_psi_dot)
        return psi_dot, V, target, distance_to_leader
