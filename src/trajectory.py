import numpy as np


def generate_l_shape_waypoints(s1=15.0, s2=15.0, origin=(0.0, 0.0)):
 
    ox, oy = origin
    waypoints = [
        (ox, oy),
        (ox + s1, oy),
        (ox + s1, oy + s2),
    ]
    return waypoints


def path_segments(waypoints):
    segments = []
    for i in range(len(waypoints) - 1):
        segments.append((np.array(waypoints[i], dtype=float),
                          np.array(waypoints[i + 1], dtype=float)))
    return segments


def dense_path_points(waypoints, resolution=0.05):

    pts = []
    for (p0, p1) in path_segments(waypoints):
        length = np.linalg.norm(p1 - p0)
        n = max(int(length / resolution), 1)
        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append(p0 + t * (p1 - p0))
    pts.append(np.array(waypoints[-1], dtype=float))
    return np.array(pts)
