import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt

from PingPong3d import PingPongBat3, PingPongBall3, PingPongMgr3
from ServeBall3d import ServeBallRobot, ServeBallBatRobot, convertData2Ball, convertData2Bat;


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc1 = nn.Linear(6, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def judge(ball, bat, f):
    mgr = PingPongMgr3(ball)
    mgr.bats[1] = bat
    mgr.start()
    return f(mgr.ball)


def getScore(ball):
    if math.fabs(ball.x) <= 0.7625 and -1.37 < ball.y <= 0:
        return math.fabs(ball.x / 0.7625) + math.fabs(ball.y / 1.37)
    else:
        return -1


if __name__ == '__main__':
    decay = 0.1
    datafile = 'data/catch-10w.txt'
    modelfile = 'model/model-10w.bin'
    r = ServeBallBatRobot(datafile)
    l = [[], []]
    score = [[], [], []]
    n = net()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(n.parameters(), lr=0.1)
    print("start learning")
    for i in range(int(len(r.lines) * 0.8)):
        optimizer.zero_grad()
        b, bat, score_target = r.generateNumbers(i)
        out = n(torch.tensor(b))

        # calculate score of actual and network fit
        score[0].append(i)
        score[1].append(score_target)
        out_d = out.detach().numpy()
        ball = convertData2Ball(b, decay)
        bat_predict = convertData2Bat(out_d)
        score[2].append(judge(ball, bat_predict, getScore))

        # calculate loss
        loss = criterion(out, torch.tensor(bat))
        # print(loss.item())
        l[0].append(i)
        l[1].append(loss.item())
        loss.backward()
        optimizer.step()

    # plot score
    print("plotting score")
    plt.scatter(score[0], score[1], label="actual", s=5)
    plt.scatter(score[0], score[2], label='dynamic', s=5)
    plt.ylabel('score')
    plt.legend()
    plt.show()

    print("plotting loss")
    plt.plot(l[0], l[1])
    plt.ylabel("Loss")
    plt.show()
    print('start saving model')
    with open(modelfile, 'wb') as f:
        pickle.dump(n, f)

    print("start testing")
    l = [[], []]
    score = [[], [], []]
    for i in range(int(len(r.lines) * 0.8), len(r.lines) - 1):
        b, bat, score_target = r.generateNumbers(i)
        out = n(torch.tensor(b))

        # calculate score of actual and network fit
        score[0].append(i)
        score[1].append(score_target)
        out_d = out.detach().numpy()
        ball = convertData2Ball(b,decay)
        bat_predict = convertData2Bat(out_d)
        score[2].append(judge(ball, bat_predict, getScore))

        # calculate loss
        loss = criterion(out, torch.tensor(bat))
        # print(loss.item())
        l[0].append(i)
        l[1].append(loss.item())

    # plot scores
    plt.scatter(score[0], score[1], label="target", s=5)
    plt.scatter(score[0], score[2], label='dynamic', s=5)
    plt.ylabel('score')
    plt.legend()
    plt.show()

    plt.close()
    plt.plot(l[0], l[1])
    plt.ylabel("Loss")
    plt.show()
