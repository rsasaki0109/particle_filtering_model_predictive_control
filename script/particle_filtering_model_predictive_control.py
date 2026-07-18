import numpy as np
import math
import matplotlib.pyplot as plt

# ==== MPC / Vehicle params ====
DT = 0.1             # [s] time step
L = 2.9              # [m] wheelbase
V_REF = 10.0 / 3.6   # [m/s] desired speed

# ==== PF-MPC params ====
H = 8                # horizon length
NP = 500             # number of particles
NTh = NP / 2.0       # resample threshold

# noise std
SIG_A = 0.5          # acceleration noise
SIG_W = np.deg2rad(5.0)  # yawrate noise

# bounds
A_MIN, A_MAX = -1.0, 1.0
W_MIN, W_MAX = np.deg2rad(-30), np.deg2rad(30)

# weights for cost
W_CTE = 1.0
W_YAW = 0.5
W_VEL = 0.2
W_DU  = 0.1
W_U   = 0.01

show_animation = True


class State:
    """Vehicle state"""
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def motion_model(x, u):
    """
    Kinematic bicycle model (x = [x, y, yaw, v])
    u = [accel, yawrate] with shape (2,1)
    """
    x_new = np.zeros_like(x)
    x_new[0, 0] = x[0, 0] + x[3, 0] * math.cos(x[2, 0]) * DT
    x_new[1, 0] = x[1, 0] + x[3, 0] * math.sin(x[2, 0]) * DT
    x_new[2, 0] = x[2, 0] + u[1, 0] * DT
    x_new[3, 0] = x[3, 0] + u[0, 0] * DT
    return x_new


def wrap_pi(angle):
    """Wrap angle to [-pi, pi]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def nearest_projection_limited(x, y, cx, cy, start_idx, max_search=10):
    """
    Search nearest course point around start_idx
    """
    search_range = range(start_idx, min(start_idx + max_search, len(cx)))
    dx = np.array([cx[i] for i in search_range]) - x
    dy = np.array([cy[i] for i in search_range]) - y
    dist2 = dx**2 + dy**2
    ind_local = int(np.argmin(dist2))
    ind = start_idx + ind_local
    path_yaw = math.atan2(cy[min(ind+1, len(cy)-1)] - cy[ind],
                          cx[min(ind+1, len(cx)-1)] - cx[ind])
    e_y = math.sqrt(dist2[ind_local])
    return ind, e_y, path_yaw


def low_variance_resampling(px, pw):
    """Low-variance resampling"""
    Neff = 1.0 / np.sum(pw**2)
    if Neff < NTh:
        positions = (np.arange(NP) + np.random.rand()) / NP
        indexes = np.zeros(NP, dtype=int)
        cumsum = np.cumsum(pw[0])
        i, j = 0, 0
        while i < NP:
            if positions[i] < cumsum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        px = px[:, indexes]
        pw = np.ones((1, NP)) / NP
    return px, pw


def pf_mpc_step(state, cx, cy, px, pw, u_prev, hint):
    pu = np.zeros((2, NP))
    costs = np.zeros(NP)
    for ip in range(NP):
        xp = np.array([px[:, ip]]).T
        u_last = u_prev.copy()
        J = 0.0
        local_hint = hint
        for i in range(H):
            a = np.clip(u_last[0, 0] + np.random.randn()*SIG_A, A_MIN, A_MAX)
            w = np.clip(u_last[1, 0] + np.random.randn()*SIG_W, W_MIN, W_MAX)
            u = np.array([[a], [w]])  # always shape (2,1)
            xp = motion_model(xp, u)
            idx, e_y, path_yaw = nearest_projection_limited(
                float(xp[0, 0]), float(xp[1, 0]), cx, cy, local_hint)
            local_hint = idx
            e_yaw = wrap_pi(float(xp[2, 0]) - path_yaw)
            e_v = float(xp[3, 0]) - V_REF
            du_a = a - u_last[0, 0]
            du_w = w - u_last[1, 0]
            reg = W_DU*(du_a**2 + du_w**2) + W_U*(a**2 + w**2)
            J += W_CTE*(e_y**2) + W_YAW*(e_yaw**2) + W_VEL*(e_v**2) + reg
            if i == 0:
                px[:, ip] = xp[:, 0]
                pu[:, ip] = u[:, 0]
            u_last = u
        costs[ip] = J
    # weight update
    Jmin = np.min(costs)
    w = np.exp(-0.5*(costs - Jmin))
    wsum = np.sum(w)
    pw[0, :] = w/wsum if wsum > 1e-12 else 1.0/NP
    xEst = px.dot(pw.T)
    state.x, state.y, state.yaw, state.v = [float(xEst[i, 0]) for i in range(4)]
    px, pw = low_variance_resampling(px, pw)
    i_best = int(np.argmax(pw))
    # get target point for best particle
    best_idx, _, _ = nearest_projection_limited(
        px[0, i_best], px[1, i_best], cx, cy, hint)
    target_point = (cx[best_idx], cy[best_idx])
    return state, px, pw, pu[:, i_best].reshape(2, 1), target_point, best_idx


def main():
    # create target course
    cx = np.arange(0, 50, 0.5)
    cy = [math.sin(ix/5.0) * ix/2.0 for ix in cx]

    # initial state
    state = State(x=0.0, y=-3.0, yaw=0.0, v=0.0)
    px = np.zeros((4, NP))
    px[1, :] = state.y
    pw = np.ones((1, NP)) / NP
    hint_idx = 0

    u_prev = np.array([[0.0], [0.0]])  # always shape (2,1)

    x_traj, y_traj, yaw_traj, v_traj, t_traj = [], [], [], [], []
    target_vis = None

    T_max = 60.0
    time = 0.0

    while time <= T_max and hint_idx < len(cx) - H:
        state, px, pw, u_prev, target_vis, hint_idx = pf_mpc_step(
            state, cx, cy, px, pw, u_prev, hint_idx)

        time += DT
        x_traj.append(state.x)
        y_traj.append(state.y)
        yaw_traj.append(state.yaw)
        v_traj.append(state.v)
        t_traj.append(time)

        if show_animation:
            plt.cla()
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x_traj, y_traj, "-b", label="trajectory")
            plt.plot(px[0, :], px[1, :], ".y", alpha=0.3, label="particles")
            if target_vis:
                plt.plot(target_vis[0], target_vis[1], "og", markersize=8, label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title(f"Speed[km/h]: {state.v*3.6:.2f}")
            plt.legend()
            plt.pause(0.001)

    # plot result
    if show_animation:
        plt.figure()
        plt.plot(cx, cy, "-r", label="course")
        plt.plot(x_traj, y_traj, "-b", label="trajectory")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        plt.figure()
        plt.plot(t_traj, [v*3.6 for v in v_traj], "-r")
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [km/h]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    print("PF-MPC with CTE+Yaw+Vel cost and target visualization")
    main()
