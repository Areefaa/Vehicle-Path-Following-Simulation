import numpy as np
from src.kinematics import wrap_angle


def closest_point_on_segment(p, a, b):
    ab = b - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-9:
        return a, 0.0
    t = np.dot(p - a, ab) / ab_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    return a + t_clamped * ab, t_clamped


class L1Guidance:
    def __init__(self, segments, L1=3.0, V=3.0, max_psi_dot=np.deg2rad(60)):

        self.segments = segments

        best_idx = self._current_seg_idx
        best_dist = np.inf
        best_proj = None
        best_t = 0.0

        search_range = range(self._current_seg_idx, len(self.segments))
        for idx in search_range:
            a, b = self.segments[idx]
            proj, t = closest_point_on_segment(pos, a, b)
            d = np.linalg.norm(pos - proj)
            if d < best_dist:
                best_dist = d
                best_idx = idx
                best_proj = proj
                best_t = t

        self._current_seg_idx = best_idx

        remaining = self.L1
        idx = best_idx
        a, b = self.segments[idx]
        seg_vec = b - a
        seg_len = np.linalg.norm(seg_vec)
        pos_along = best_t * seg_len

        while True:
            dist_to_seg_end = seg_len - pos_along
            if remaining <= dist_to_seg_end:
                target = a + (seg_vec / seg_len) * (pos_along + remaining)
                return target
            remaining -= dist_to_seg_end
            idx += 1
            if idx >= len(self.segments):
                return self.segments[-1][1]
            a, b = self.segments[idx]
            seg_vec = b - a
            seg_len = np.linalg.norm(seg_vec)
            pos_along = 0.0

    def compute_control(self, state):
     
        pos = state.position()
        target = self._find_lookahead_point(pos)

        l1_vec = target - pos
        l1_dist = np.linalg.norm(l1_vec)
        if l1_dist < 1e-6:
            return 0.0, self.V, target

        l1_angle = np.arctan2(l1_vec[1], l1_vec[0])
        eta = wrap_angle(l1_angle - state.psi)

        a_s = (2 * self.V ** 2 / self.L1) * np.sin(eta)
        psi_dot = a_s / self.V

        psi_dot = np.clip(psi_dot, -self.max_psi_dot, self.max_psi_dot)
        return psi_dot, self.V, target

    def is_finished(self, state, capture_radius=0.5):
        last_wp = self.segments[-1][1]
        return np.linalg.norm(state.position() - last_wp) < capture_radius
