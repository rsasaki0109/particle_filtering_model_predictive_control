import numpy as np
import math
import matplotlib.pyplot as plt

# ===== Params =====
DT   = 0.1
NP   = 400              # 粒子数を増やして安定化
H    = 6                # 先読み長（追従重視）
k    = 0.1
Lfc  = 2.0
TMAX = 50.0

# 速度目標（一定でOK。必要なら曲率依存にしても良い）
V_REF = 5.0  # [m/s]

# 乱数サンプリング幅（前回入力の周りにサンプル）
SIG_A = 0.6               # m/s^2
SIG_W = np.deg2rad(12.0)  # rad/s

# 入力・速度制限（暴走抑制）
A_MIN, A_MAX = -2.5, 2.5
W_MIN, W_MAX = -1.0, 1.0
V_MIN, V_MAX = 0.0, 12.0

# コスト重み（横ずれ・向き・速度・入力変化）
W_CTE   = 1.0
W_YAW   = 0.35
W_VEL   = 0.15
W_DU    = 0.02

np.random.seed(0)  # 再現性

# ===== Helpers =====
def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

class State:
    def __init__(self, x=0.0, y=-3.0, yaw=0.0, v=0.0):
        self.x=float(x); self.y=float(y); self.yaw=float(yaw); self.v=float(v)

def target_trajectory(ix):
    return math.sin(ix / 5.0) * ix / 2.0

def build_path(x_end=50.0, dx=0.1):
    cx = np.arange(0.0, x_end, dx)
    cy = np.array([target_trajectory(ix) for ix in cx])
    return cx, cy

def nearest_on_path(x, y, cx, cy):
    d = (cx-x)**2 + (cy-y)**2
    i = int(np.argmin(d))
    i2 = min(i+1, len(cx)-1)
    dx = cx[i2]-cx[i]; dy = cy[i2]-cy[i]
    path_yaw = math.atan2(dy, dx) if (dx!=0 or dy!=0) else 0.0
    # 符号付き横ずれ：左正
    sign = np.sign(dx*(y-cy[i]) - dy*(x-cx[i]))
    e_y  = float(sign * math.hypot(x-cx[i], y-cy[i]))
    return i, e_y, path_yaw

def calc_target_index(state, cx, cy, old_idx=None):
    if old_idx is None:
        d = [math.hypot(state.x - ix, state.y - iy) for ix, iy in zip(cx, cy)]
        ind = int(np.argmin(d))
    else:
        ind = int(old_idx)
        dst = math.hypot(state.x - cx[ind], state.y - cy[ind])
        while True:
            nxt = ind + 1 if (ind + 1) < len(cx) else ind
            dst2 = math.hypot(state.x - cx[nxt], state.y - cy[nxt])
            if dst < dst2 or nxt == ind: break
            ind, dst = nxt, dst2
    # 先行点
    Lacc = 0.0; Lf = k * state.v + Lfc
    while Lf > Lacc and (ind + 1) < len(cx):
        Lacc = math.hypot(cx[ind]-state.x, cy[ind]-state.y)
        ind += 1
    return ind

# ===== Models =====
def motion_model(x, u):
    """ x=[x,y,yaw,v]^T, u=[a,w]^T """
    a=float(u[0,0]); w=float(u[1,0]); v=float(x[3,0])
    nx = np.zeros_like(x)
    nx[0,0] = float(x[0,0] + v * math.cos(x[2,0]) * DT)
    nx[1,0] = float(x[1,0] + v * math.sin(x[2,0]) * DT)
    nx[2,0] = wrap_pi(float(x[2,0] + w * DT))
    nx[3,0] = float(np.clip(v + a * DT, V_MIN, V_MAX))
    return nx

def low_variance_resampling(px, pw):
    NP_local = px.shape[1]
    Neff = 1.0 / (pw.dot(pw.T))[0,0]
    if Neff < NP_local/2.0:
        w = pw[0].copy(); w/=w.sum()
        wc = np.cumsum(w)
        r0 = np.random.rand() / NP_local
        pos = r0 + (np.arange(NP_local) / NP_local)
        inds = np.zeros(NP_local, dtype=int); i=j=0
        while i<NP_local and j<NP_local:
            if pos[i] < wc[j]: inds[i]=j; i+=1
            else: j+=1
        px = px[:, inds]
        pw = np.zeros((1, NP_local)) + 1.0 / NP_local
    return px, pw

# ===== PF-MPC (追従コストで評価) =====
def pf_mpc_step(state, cx, cy, px, pw, u_prev):
    """
    粒子ごとに H ステップの将来をサンプルし、累積コストで重み付け。
    最重粒子の「最初の入力」を次入力に採択。
    """
    pu = np.zeros((2, NP))
    costs = np.zeros(NP)

    for ip in range(NP):
        xp = np.array([px[:, ip]]).T
        u_last = u_prev.copy()
        J = 0.0

        for i in range(H):
            # 前回入力の周りでサンプル
            a = float(u_last[0] + np.random.randn()*SIG_A)
            w = float(u_last[1] + np.random.randn()*SIG_W)
            a = float(np.clip(a, A_MIN, A_MAX))
            w = float(np.clip(w, W_MIN, W_MAX))
            u = np.array([[a, w]]).T

            xp = motion_model(xp, u)

            # 誤差（横ずれ・向き・速度）
            _, e_y, path_yaw = nearest_on_path(float(xp[0,0]), float(xp[1,0]), cx, cy)
            e_yaw = wrap_pi(float(xp[2,0]) - path_yaw)
            e_v   = float(xp[3,0]) - V_REF

            # 入力正則化（前ステップからの変化も罰則）
            du_a = float(u[0,0] - u_last[0,0])
            du_w = float(u[1,0] - u_last[1,0])

            J += W_CTE*(e_y**2) + W_YAW*(e_yaw**2) + W_VEL*(e_v**2) + W_DU*(du_a**2 + du_w**2)

            if i == 0:
                px[:, ip] = xp[:, 0]
                pu[:, ip] = u[:, 0]

            u_last = u

        costs[ip] = J

    # コスト -> 重み
    Jmin = costs.min()
    # 数値安定化のため基準引き（softmax的に）
    w = np.exp(-0.5 * (costs - Jmin))
    w_sum = w.sum()
    if w_sum <= 1e-12 or not np.isfinite(w_sum):
        pw[:] = 1.0 / NP
    else:
        pw[0, :] = w / w_sum

    # 推定（加重平均）
    xEst = px.dot(pw.T)
    state.x = float(xEst[0,0]); state.y = float(xEst[1,0])
    state.yaw = float(xEst[2,0]); state.v = float(xEst[3,0])

    # リサンプリング
    px, pw = low_variance_resampling(px, pw)

    # 最良入力（最重粒子の最初の入力）
    i_best = int(np.argmax(pw))
    u_next = pu[:, i_best].reshape(2,1)
    return state, px, pw, u_next

# ===== Main =====
def main(show_animation=True):
    cx, cy = build_path(50.0, 0.1)
    last_idx = len(cx) - 1

    state = State(x=0.0, y=-3.0, yaw=0.0, v=0.0)

    px = np.zeros((4, NP))
    px[0,:]=state.x; px[1,:]=state.y; px[2,:]=state.yaw; px[3,:]=state.v
    pw = np.zeros((1, NP)) + 1.0/NP

    u = np.array([[0.5, 0.0]]).T  # 軽く前へ
    t=0.0
    xs,ys,vs,ts=[state.x],[state.y],[state.v],[0.0]
    target_ind = calc_target_index(state, cx, cy, None)

    while t<=TMAX and (last_idx - H) > target_ind:
        state, px, pw, u = pf_mpc_step(state, cx, cy, px, pw, u)

        # 可視化・ログ
        t += DT
        xs.append(state.x); ys.append(state.y); vs.append(state.v); ts.append(t)
        target_ind = calc_target_index(state, cx, cy, target_ind)

        if show_animation:
            plt.cla()
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(xs, ys, "-b", label="trajectory")
            plt.scatter(px[0,:], px[1,:], s=6, c="y")
            plt.axis("equal"); plt.grid(True); plt.legend()
            plt.title(f"v={state.v*3.6:.1f} km/h"); plt.pause(0.001)

    if show_animation:
        plt.figure()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(xs, ys, "-b", label="trajectory")
        plt.axis("equal"); plt.grid(True); plt.legend()
        plt.xlabel("x [m]"); plt.ylabel("y [m]")

        plt.figure()
        plt.plot(ts, [v*3.6 for v in vs], "-")
        plt.xlabel("Time [s]"); plt.ylabel("Speed [km/h]")
        plt.grid(True); plt.title("Speed")
        plt.show()

if __name__ == "__main__":
    main(show_animation=True)
