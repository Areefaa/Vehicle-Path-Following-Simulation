import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class VehicleState:
    def __init__(self, x=0.0, y=0.0, psi=0.0):
        self.x = x
        self.y = y
        self.psi = psi

    def as_array(self):
        return np.array([self.x, self.y, self.psi])

    def position(self):
        return np.array([self.x, self.y])


def step(state: VehicleState, V, psi_dot, dt, max_psi_dot=None):

    if max_psi_dot is not None:
        psi_dot = np.clip(psi_dot, -max_psi_dot, max_psi_dot)

    x_new = state.x + V * np.cos(state.psi) * dt
    y_new = state.y + V * np.sin(state.psi) * dt
    psi_new = wrap_angle(state.psi + psi_dot * dt)

    return VehicleState(x_new, y_new, psi_new)
