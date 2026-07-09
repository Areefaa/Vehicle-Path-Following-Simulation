import bisect
import numpy as np

class LeaderTrail:
    def __init__(self):
        self.positions = []   
        self.cum_dist = [0.0]  

    def append(self, pos):
        pos = np.array(pos, dtype=float)
        if self.positions:
            d = np.linalg.norm(pos - self.positions[-1])
            self.cum_dist.append(self.cum_dist[-1] + d)
        self.positions.append(pos)

    def total_length(self):
        return self.cum_dist[-1] if self.cum_dist else 0.0

    def point_at_arclength_behind(self, d0):

        if not self.positions:
            raise ValueError("Trail masih kosong, belum ada posisi leader yang direkam.")

        total = self.total_length()
        target_len = max(total - d0, 0.0)

        if len(self.positions) == 1:
            return self.positions[0]

        idx = bisect.bisect_left(self.cum_dist, target_len)
        idx = min(max(idx, 1), len(self.positions) - 1)

        d_prev = self.cum_dist[idx - 1]
        d_next = self.cum_dist[idx]
        if d_next - d_prev < 1e-9:
            return self.positions[idx]

        t = (target_len - d_prev) / (d_next - d_prev)
        return self.positions[idx - 1] + t * (self.positions[idx] - self.positions[idx - 1])


def formation_speed_rule(distance_to_leader):
    if distance_to_leader >= 5.0:
        return 3.0
    elif distance_to_leader >= 4.0:
        return 1.5
    else:
        return 1.0
