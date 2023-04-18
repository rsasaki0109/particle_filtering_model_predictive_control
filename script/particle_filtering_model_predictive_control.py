import numpy as np
import math
import matplotlib.pyplot as plt

k = 0.1  # look forward gain
Lfc = 2.0  # look-ahead distance
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s]
L = 2.9  # [m] wheel base of vehicle

# Estimation parameter of PF-MPC
Q = np.diag([0.1])**2  # range error

#  Simulation parameter
Qsim = np.diag([0.2])**2
Rsim = np.diag([1.0, np.deg2rad(30.0)])**2

DT = 0.1  # time tick [s]

# pf-mpc parameter
H = 1  # horizon length
NP = 200  # Number of Particle
NTh = NP / 2.0  # Number of particle for re-sampling

old_nearest_point_index = None
show_animation = True


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def motion_model(x, u):

    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F.dot(x) + B.dot(u)

    return x

def gauss_likelihood(x, sigma):
    p = 1.0 / (2.0 * np.pi * np.linalg.det(sigma))**0.5 * \
        np.exp(-0.5 * x.T @ np.linalg.inv(sigma) @ x)

    return p

def resampling(px, pw):
    """
    low variance re-sampling
    """

    Neff = 1.0 / (pw.dot(pw.T))[0, 0]  # Effective particle number
    if Neff < NTh:
        wcum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / NP) - 1 / NP
        resampleid = base + np.random.rand(base.shape[0]) / NP

        inds = []
        ind = 0
        for ip in range(NP):
            while resampleid[ip] > wcum[ind]:
                ind += 1
            inds.append(ind)

        inds = np.array(inds, dtype=int)  # Add this line
        px = px[:, inds]
        pw = np.zeros((1, NP)) + 1.0 / NP  # init weight

    return px, pw

def target_trajectory(ix):
    t = math.sin(ix / 5.0) * ix / 2.0

    return t

def refrence_trajectory(state, x, ty):
    Ts = 1
    s = (ty - state.y) * (1 - math.exp(-(x - state.x) / Ts)) + state.y

    return s

def pf_mpc(state, cx, cy, pind, px, pw, u):
    pu = np.zeros((2, NP))  # Particle imput

    ind = calc_target_index(state, cx, cy)

    if pind >= ind:
        ind = pind

    if not (ind < len(cx)):
        ind = len(cx) - 1

    for ip in range(NP):
        xp = np.array([px[:, ip]]).T
        w = pw[0, ip]
        for i in range(H):
            #  Predict with random input sampling
            ud1 = u[0] + np.random.randn() * Rsim[0, 0] * 4
            ud2 = u[1] + np.random.randn() * Rsim[1, 1] * 4
            ud = np.array([[ud1, ud2]]).T
            xp = motion_model(xp, ud)
            if i == 0:
                px[:, ip] = xp[:, 0].reshape(-1,)
            if i == 1:
                pu[:, ip] = ud[:, 0]

        ty = target_trajectory(xp[0, 0])
        s = refrence_trajectory(state, xp[0, 0], ty)
        dist = abs(s - xp[1, 0])
        w = gauss_likelihood(np.array([[dist]]), Q)
        pw[0, ip] = w

    pw = pw / pw.sum()  # normalize

    xEst = px.dot(pw.T)
    state.x = xEst[0]
    state.y = xEst[1]
    state.yaw = xEst[2]
    state.v = xEst[3]

    px, pw = resampling(px, pw)

    i_wmax = np.argmax(pw)

    u = pu[:, i_wmax]

    return state, ind, px, pw, u

def calc_distance(state, point_x, point_y):

    dx = state.x - point_x
    dy = state.y - point_y
    return math.sqrt(dx ** 2 + dy ** 2)

def calc_target_index(state, cx, cy):

    global old_nearest_point_index

    if old_nearest_point_index is None:
        # search nearest point index
        dx = [state.x - icx for icx in cx]
        dy = [state.y - icy for icy in cy]
        d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
        ind = d.index(min(d))
        old_nearest_point_index = ind
    else:
        ind = old_nearest_point_index
        distance_this_index = calc_distance(state, cx[ind], cy[ind])
        while True:
            ind = ind + 1 if (ind + 1) < len(cx) else ind
            distance_next_index = calc_distance(state, cx[ind], cy[ind])
            if distance_this_index < distance_next_index:
                break
            distance_this_index = distance_next_index
        old_nearest_point_index = ind

    L = 0.0

    Lf = k * state.v + Lfc

    # search look ahead target point index
    while Lf > L and (ind + 1) < len(cx):
        dx = cx[ind] - state.x
        dy = cy[ind] - state.y
        L = math.sqrt(dx ** 2 + dy ** 2)
        ind += 1

    return ind

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def main():
    #  target course
    cx = np.arange(0, 50, 0.1)
    cy = [target_trajectory(ix) for ix in cx]

    T = 100.0  # max simulation time

    # initial state
    state = State(x=-0.0, y=-3.0, yaw=0.0, v=0.0)
    px = np.zeros((4, NP))  # Particle store
    px[1, :] = state.y
    pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight

    lastIndex = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_ind = calc_target_index(state, cx, cy)

    u = np.array([[0.0, 0.0]]).T

    while T >= time and lastIndex - H > target_ind:

        state, target_ind, px, pw, u = pf_mpc(state, cx, cy, target_ind, px, pw, u)

        time = time + dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if show_animation:  # pragma: no cover
            plt.cla()
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.plot(px[0, :], px[1, :], ".y")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert lastIndex >= target_ind, "Cannot goal"

    if show_animation:  # pragma: no cover
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(t, [iv * 3.6 for iv in v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    print("Particle filtering model predictive control simulation start")
    main()
