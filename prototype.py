# %%

from typing import Any
import pandas as pd
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import jacobian

# %%

n_timesteps = 100


class Device:
    def __init__(
        self, name: str, cons_on: float, cons_off: float, state_changes: dict[int, bool]
    ):
        self.name = name
        self.cons_on = cons_on
        self.cons_off = cons_off
        self.state_changes = state_changes
        self.state = False
        """False if off, True if on"""

    def get_consumption(self, t: int, noise: bool) -> float:
        state_change = self.state_changes.get(t)
        if state_change is not None:
            self.state = state_change
        consumption_true = self.cons_on if self.state else self.cons_off
        noise_value = 0.0
        if noise:
            noise_value = np.random.normal(0, 3)
        return consumption_true + noise_value


devices = [
    Device(
        "d1", 100, 0, {10: True, 20: False, 30: True, 40: False, 50: True, 55: False}
    ),
    Device("d2", 150, 75, {15: True, 25: False}),
]


# %%

"""
State x_k: (6,)
 - d1_state
 - d1_on_consumption
 - d1_off_consumption
 - d2_state
 - d2_on_consumption
 - d2_off_consumption

J_f: Identity matrix (6x6)

Observation z_k: (3,)
 - total_consumption
 - d1_state
 - d2_state

H_k: Identity matrix (3x6)

"""

U_D1_STATE = 0
U_D2_STATE = 1

X_D1_STATE = 0
X_D1_ON_CONSUMPTION = 1
X_D1_OFF_CONSUMPTION = 2
X_D2_STATE = 3
X_D2_ON_CONSUMPTION = 4
X_D2_OFF_CONSUMPTION = 5

Z_TOTAL_CONSUMPTION = 0
Z_D1_STATE = 1
Z_D2_STATE = 2


def f(x_k1: np.ndarray, u_k: np.ndarray) -> np.ndarray:
    return np.array(
        [
            u_k[U_D1_STATE],
            x_k1[X_D1_ON_CONSUMPTION],
            x_k1[X_D1_OFF_CONSUMPTION],
            u_k[U_D2_STATE],
            x_k1[X_D2_ON_CONSUMPTION],
            x_k1[X_D2_OFF_CONSUMPTION],
        ]
    )


def h(x_k: np.ndarray) -> np.ndarray:
    if x_k[X_D1_STATE] > 0.5:
        d1_consumption = x_k[X_D1_ON_CONSUMPTION]
    else:
        d1_consumption = x_k[X_D1_OFF_CONSUMPTION]
    if x_k[X_D2_STATE] > 0.5:
        d2_consumption = x_k[X_D2_ON_CONSUMPTION]
    else:
        d2_consumption = x_k[X_D2_OFF_CONSUMPTION]
    total_consumption = d1_consumption + d2_consumption

    return np.array([total_consumption])


logs = []

x_0 = np.zeros(6)
x_0[X_D1_STATE] = 0
x_0[X_D1_ON_CONSUMPTION] = 50
x_0[X_D1_OFF_CONSUMPTION] = 0
x_0[X_D2_STATE] = 0
x_0[X_D2_ON_CONSUMPTION] = 50
x_0[X_D2_OFF_CONSUMPTION] = 0

x_0[X_D1_ON_CONSUMPTION] = 120

P_0 = np.eye(6)
P_0[X_D1_ON_CONSUMPTION, X_D1_ON_CONSUMPTION] = 300
P_0[X_D1_OFF_CONSUMPTION, X_D1_OFF_CONSUMPTION] = 0.01
P_0[X_D2_ON_CONSUMPTION, X_D2_ON_CONSUMPTION] = 300
P_0[X_D2_OFF_CONSUMPTION, X_D2_OFF_CONSUMPTION] = 50
Q = np.eye(6) * 0.1  # Process noise

noise = True

x_k1 = x_0
P_k1 = P_0

xs, Ps, z_preds = [], [], []
for k in range(n_timesteps):
    # Simulation step
    total_consumption = 0.0
    log: dict[str, Any] = {}
    for device in devices:
        device_consumption = device.get_consumption(k, noise)
        total_consumption += device_consumption
        log[f"{device.name}_state"] = device.state
        log[f"{device.name}_consumption"] = device_consumption
    log["total_consumption"] = total_consumption
    logs.append(log)

    # Observation for Kalman filter
    z_k = np.array([total_consumption])
    u_k = np.array([1.0 if d.state else 0.0 for d in devices])

    # Prediction step
    x_k_pred = f(x_k1, u_k)
    J_f = jacobian(lambda x: f(x, u_k))(x_k1)
    P_k_pred = J_f @ P_k1 @ J_f.T + Q

    # Update step
    R = np.eye(1) * 5  # Observation noise
    J_h = jacobian(h)(x_k_pred)
    K = P_k_pred @ J_h.T @ np.linalg.inv(J_h @ P_k_pred @ J_h.T + R)
    z_k_pred = h(x_k_pred)
    x_k1 = x_k_pred + K @ (z_k - z_k_pred)
    P_k1 = (np.eye(6) - K @ J_h) @ P_k_pred


    xs.append(x_k1)
    Ps.append(P_k1)
    z_preds.append(z_k_pred)

data = pd.DataFrame(logs)


fig = plt.figure(figsize=(10, 5))
gs = plt.GridSpec(1 + len(devices), 1)

ax = fig.add_subplot(gs[0, 0])
ax.step(data.index, data["total_consumption"], where="post", color="black")
ax.step(data.index, [z_k[0] for z_k in z_preds], where="post", color="orange")
ax.set(title="Total Consumption")

for i, device in enumerate(devices):
    ax = fig.add_subplot(gs[1 + i, 0], sharex=ax, sharey=ax)
    ax.step(data.index, data[f"{device.name}_consumption"], where="post")
    ax2 = ax.twinx()
    ax2.step(
        data.index,
        data[f"{device.name}_state"],
        where="post",
        color="black",
        linestyle="--",
    )

    d_cons = np.zeros(n_timesteps)
    for k in range(n_timesteps):
        d_cons[k] = (
            xs[k][X_D1_STATE + i * 3] * xs[k][X_D1_ON_CONSUMPTION + i * 3]
            + (1 - xs[k][X_D1_STATE + i * 3]) * xs[k][X_D1_OFF_CONSUMPTION + i * 3]
        )
    ax.step(data.index, d_cons, where="post", color="orange")

    ax.set(title=f"{device.name} Consumption")
    ax.set(title=f"{device.name} Consumption")

k = 60
ax.axvline(k, color="red", linestyle="--")
x_true = np.zeros(6)
x_true[X_D1_STATE] = 0
x_true[X_D1_ON_CONSUMPTION] = 120
x_true[X_D1_OFF_CONSUMPTION] = 0
x_true[X_D2_STATE] = 0
x_true[X_D2_ON_CONSUMPTION] = 150
x_true[X_D2_OFF_CONSUMPTION] = 0
print(" ".join([f"{x:7.3f}" for x in x_true]))
print(" ".join([f"{x:7.3f}" for x in xs[k]]))
print()

for i in range(6):
    print(" ".join([f"{x:7.3f}" for x in Ps[k][i, :]]))

fig.tight_layout()


# %%
jacobian(h)(x_k)

# %%
