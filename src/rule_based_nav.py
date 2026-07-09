import numpy as np
from src.kinematics import wrap_angle


class RuleBasedWaypointNav:
    def __init__(self, waypoints, capture_radius=1.0,
                 kp_heading=1.5, max_psi_dot=np.deg2rad(60),
                 use_speed_tiers=True, constant_V=3.0):

        self.waypoints = [np.array(wp, dtype=float) for wp in waypoints]
        self.capture_radius = capture_radius
        self.kp_heading = kp_heading
        self.max_psi_dot = max_psi_dot
        self.use_speed_tiers = use_speed_tiers
        self.constant_V = constant_V
        self._target_idx = 1 

    def _speed_rule(self, distance):
        if not self.use_speed_tiers:
            return self.constant_V
        if distance >= 5.0:
            return 3.0
        elif distance >= 4.0:
            return 1.5
        else:
            return 1.0

    def compute_control(self, state):

        pos = state.position()
        target = self.waypoints[min(self._target_idx, len(self.waypoints) - 1)]
        vec = target - pos
        distance = np.linalg.norm(vec)

        if distance < self.capture_radius and self._target_idx < len(self.waypoints) - 1:
            self._target_idx += 1
            target = self.waypoints[self._target_idx]
            vec = target - pos
            distance = np.linalg.norm(vec)

        bearing = np.arctan2(vec[1], vec[0])
        heading_error = wrap_angle(bearing - state.psi)

        psi_dot = self.kp_heading * heading_error
        psi_dot = np.clip(psi_dot, -self.max_psi_dot, self.max_psi_dot)

        V = self._speed_rule(distance)
        return psi_dot, V, target

    def is_finished(self, state, capture_radius=0.5):
        last_wp = self.waypoints[-1]
        return (self._target_idx == len(self.waypoints) - 1 and
                np.linalg.norm(state.position() - last_wp) < capture_radius)
