from collections import namedtuple

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from PingPong3d import PingPongBat3, PingPongBall3, PingPongMgr3, InterceptPoint3
from ServeBall3d import ServeBallRobot, ServeBallBatRobot, convertData2Ball, convertData2Bat


class PingPongEnv:
    def __init__(self, datafile, decay):
        self.mgr = PingPongMgr3(None)
        self.bot = ServeBallRobot(datafile)
        self.decay = decay

    def reset(self):
        self.mgr.ball = self.bot.generateBall(self.decay)

    def getReward(self, action, judge):
        bat = convertData2Bat(action.cpu().detach().numpy())
        self.mgr.bats[1] = bat
        self.mgr.start()
        return judge(self.mgr.ball)

    def getState(self):
        b = self.mgr.ball
        v = b.v
        return [b.x, b.y, b.z, v.x, v.y, v.z]


Transition = namedtuple('Transition', ('state', 'action', 'reward'))


# class ReplayMemory:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0
#
#     def push(self, *args):
#         if len(self.memory) < self.capacity:
#             # place holder
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity
#
#     def random(self):
#         return random.choice(self.memory)
#
#     def __len__(self):
#         return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(6, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc_V = nn.Linear(100, 1)
        self.fc_mu = nn.Linear(100, action_dim)
        self.fc_L0 = nn.Linear(100, action_dim * (action_dim + 1) // 2)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        V = self.fc_V(s)
        mu = self.fc_mu(s)
        L0 = self.fc_L0(s)
        return V, mu, L0


def select_action(state):
    global step_done
    sample = random.random()
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-step_done / EPS_DECAY)
    if sample < eps:
        # generate random action
        ball = convertData2Ball(state, decay)
        x, y, z, _ = InterceptPoint3(ball, random.uniform(0, 1))
        theta = random.random() * np.pi
        phi = random.random() * np.pi
        vx = random.uniform(-0.5, 0.5)
        vy = -random.random() / 2
        vz = random.uniform(-0.5, 0.5)
        return torch.tensor([x, y, z, vx, vy, vz, theta, phi])
    else:
        with torch.no_grad():
            _, action, _ = policy_net(torch.tensor(state))
            return action


def plot_score():
    global scores
    plt.figure(2)
    plt.clf()
    scores_t = torch.tensor(scores, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('score')
    plt.plot(scores_t.numpy())
    # Take 100 episode averages and plot them too
    if len(scores_t) >= 100:
        # get all slice by unfold
        # calculate means by dimension 1
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model(state, reward, action):
    V, mu, L0 = policy_net(torch.tensor(state))
    L = torch.zeros(action_dim, action_dim)
    index = 0
    for i in range(action_dim):
        for j in range(i + 1):
            L[i][j] = L0[index]
            index += 1
    for i in range(action_dim):
        L[i][i] = torch.exp(L[i][i])
    P = torch.transpose(L, 0, 1) * L
    z = action - mu
    advantage = z @ (P @ z.view(action_dim, 1))
    state_action_value = advantage + V
    loss = F.smooth_l1_loss(state_action_value, torch.tensor([reward], dtype=torch.float))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def CornerJudge(ball):
    if len(ball.TableBouncePoint[1]) >= 2 and -1.37 < ball.TableBouncePoint[1][1] < 0:
        return np.fabs(ball.TableBouncePoint[0][1]) / 0.7625 + np.fabs(ball.TableBouncePoint[1][1]) / 1.37 + 0.1
    elif len(ball.TableBouncePoint[1]) == 1 and ball.v.y < 0:
        return 0.1
    elif len(ball.TableBouncePoint[1]) == 1 and ball.v.y > 0 or len(ball.TableBouncePoint[1]) >= 2 and 0 < \
            ball.TableBouncePoint[1][1] < 1.37:
        return 0.
    else:
        return 0.


action_dim = 8
state_dim = 6

# train params
# BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 100

policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
# 不启用 BatchNormalization 和 Dropout
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
# memory = ReplayMemory(10000)

step_done = 0
decay = 0.1

scores = []

num_episodes = 10000
output = "DQN.log"
envfile = "data/throw-10w.txt"


env = PingPongEnv(envfile, decay)
f = open(output,'w')
for i_episode in range(num_episodes):
    env.reset()
    state: list = env.getState()
    action = select_action(state)
    reward = env.getReward(action, CornerJudge)
    optimize_model(state, reward, action)

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    scores.append(reward)
    f.write(f"{reward}\n")
    plot_score()

f.close()