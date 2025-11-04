import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ====== CartPole 環境の簡易模倣 ======
class SimpleCartPole:
    def __init__(self):
        self.x = 0.0         # 位置
        self.v = 0.0         # 速度
        self.theta = 0.05    # 棒の角度
        self.omega = 0.0     # 角速度
        self.dt = 0.02
        self.gravity = 9.8
        self.force_mag = 10.0
        self.length = 0.5
        self.max_steps = 300
        self.reset()

    def reset(self):
        self.x, self.v, self.theta, self.omega = np.random.uniform(-0.05, 0.05, 4)
        self.t = 0
        return np.array([self.x, self.v, self.theta, self.omega], dtype=np.float32)

    def step(self, action):
        force = self.force_mag * (1 if action == 1 else -1)
        costheta, sintheta = np.cos(self.theta), np.sin(self.theta)
        temp = (force + self.length * self.omega**2 * sintheta) / 1.0
        theta_acc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - 0.1 * costheta**2))
        x_acc = temp - 0.1 * theta_acc * costheta
        self.x += self.dt * self.v
        self.v += self.dt * x_acc
        self.theta += self.dt * self.omega
        self.omega += self.dt * theta_acc
        self.t += 1
        done = abs(self.x) > 2.4 or abs(self.theta) > 0.2 or self.t >= self.max_steps
        reward = 1.0 if not done else 0.0
        return np.array([self.x, self.v, self.theta, self.omega], dtype=np.float32), reward, done

# ====== 学習済みモデルのロード ======
from cart1 import PolicyNet  # 同一プロジェクト内にあると仮定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PolicyNet(4, 2).to(device)
model.load_state_dict(torch.load("ppo_cart1.pth", map_location=device))
model.eval()

# ====== アニメーションセットアップ ======
env = SimpleCartPole()
state = env.reset()

fig, ax = plt.subplots(figsize=(6,3))
cart, = ax.plot([], [], "ks", markersize=20)
pole, = ax.plot([], [], "r-", linewidth=4)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.5, 1.5)
ax.set_title("PPO Policy Visualization (CartPole)")

def animate(i):
    global state
    s = torch.tensor(state, dtype=torch.float32, device=device)
    logits = model(s)
    action = torch.argmax(logits).item()
    state, _, done = env.step(action)
    x = env.x
    theta = env.theta
    # cart
    cart.set_data([x], [0])
    # pole tip
    pole.set_data([x, x + np.sin(theta)], [0, np.cos(theta)])
    if done:
        state = env.reset()
    return cart, pole

ani = animation.FuncAnimation(fig, animate, frames=500, interval=40, blit=True)
plt.show()
