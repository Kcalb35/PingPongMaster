import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt

from PingPong3d import PingPongBat3, PingPongBall3, PingPongMgr3
from ServeBall3d import ServeBallRobot, ServeBallBatRobot, convertData2Ball, convertData2Bat

from tqdm import tqdm


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


class netE1(nn.Module):
    def __init__(self):
        super(netE1, self).__init__()
        self.fc1 = nn.Linear(6, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class netE2(nn.Module):
    def __init__(self):
        super(netE2, self).__init__()
        self.fc1 = nn.Linear(6, 30)
        self.fc2 = nn.Linear(30, 50)
        self.fc3 = nn.Linear(50, 30)
        self.fc4 = nn.Linear(30, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def judge(ball, bat, f):
    mgr = PingPongMgr3(ball)
    mgr.bats[1] = bat
    mgr.start()
    return f(mgr.ball)


def getScore(ball):
    li = ball.TableBouncePoint
    if len(li[0]) <= 1:
        return -2
    else:
        if -0.7625 <= li[0][1] <= 0.7625 and -1.37 < li[1][1] <= 0:
            return math.fabs(li[0][1] / 0.7625) + math.fabs(li[1][1] / 1.37)
    return -1


if __name__ == '__main__':
    n = net()
    decay = 0.1
    exp_code = 'h7'
    pic_path = f"pics/{exp_code}"
    datafile = 'data/catch-more20-1'
    modelfile = f'model/model-{exp_code}.bin'
    scorefile = f'score/score_{exp_code}'
    r = ServeBallBatRobot(datafile)
    l = [[], []]
    score = [[], [], []]
    criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(n.parameters(), lr=0.1)
    print(f"{exp_code}-{datafile}")
    print("start learning")
    for i in tqdm(range(int(len(r.lines) * 0.8))):
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
    plt.scatter(score[1], score[2], s=5)
    plt.title(f"{exp_code}-train")
    plt.xlabel("actual")
    plt.ylabel('neural network')
    plt.savefig(pic_path + "_train_score.jpg", dpi=200)
    plt.close()
    with open(scorefile + "_train", 'wb') as f:
        pickle.dump(score, f)

    print("plotting loss")
    plt.plot(l[0], l[1])
    plt.ylabel("Loss")
    plt.savefig(pic_path + "_train_loss.jpg", dpi=200)
    plt.close()

    print('start saving model')
    with open(modelfile, 'wb') as f:
        pickle.dump(n, f)

    print("start testing")
    l = [[], []]
    score = [[], [], []]
    for i in tqdm(range(int(len(r.lines) * 0.8), len(r.lines) - 1)):
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

    # plot scores
    plt.title(f"{exp_code}-test")
    plt.scatter(score[1], score[2], s=5)
    plt.xlabel('actual')
    plt.ylabel('neural network')
    plt.savefig(pic_path + "_test_score.jpg", dpi=200)
    plt.close()

    with open(scorefile + "_test", 'wb') as f:
        pickle.dump(score, f)

    # plot test loss
    plt.close()
    plt.plot(l[0], l[1])
    plt.ylabel("Loss")
    plt.savefig(pic_path + "_test_loss.jpg", dpi=200)
    plt.close()
