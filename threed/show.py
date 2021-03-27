import matplotlib.pyplot as plt

from PingPong3d import PingPongMgr3, InterceptPoint3, PingPongBat3, DroppingStatus, PingPongBall3
from ServeBall3d import ServeBallRobot, ServeBallBatRobot

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    bot = ServeBallRobot('test/testthrow.txt')
    r = ServeBallBatRobot('test/test-catch.txt')

    b, bat, _ = r.generate(0.1)

    # b = bot.generateBallIndex(0, 0.1)

    # bat = PingPongBat3(0.20988, 1.466444, 0.143454, 0.25764,-0.45152,0.428729, 1.38, 2.349)
    # with open('model.bin', 'rb') as f:
    #     n = pickle.load(f)
    # bat = n.forward(torch.tensor([b.x, b.y, b.z, b.v.x, b.v.y, b.v.z])).tolist()
    # bat = PingPongBat3(bat[0], bat[1], bat[2], bat[3], bat[4], bat[5], bat[6], bat[7])

    # bat1 = PingPongBat3(p[0], p[1]+b.radius, 0, 0, 0, 0, math.pi / 2, math.pi / 2)
    # ax.scatter(p[0], p[1], p[2], color='r')
    # ax.scatter(d[0], d[1], 0.02, color='r')
    mgr = PingPongMgr3(b)
    mgr.bats[1] = bat
    mgr.start()
    mgr.show(ax)
    ax.scatter([bat.x], [bat.y], [bat.z], color='r')
    plt.show()
