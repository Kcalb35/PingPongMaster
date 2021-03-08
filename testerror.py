from PingPong import *

if __name__ == '__main__':
    robot = ServeBallRobot()
    target = -1
    sum = 0
    li = []
    while True:
        ball = robot.GenerateBall()
        bat = CatchBall_Bat(ball, DroppingPoint(ball)[0] + 0.1, target)
        mgr = PingPongManager(ball)
        mgr.tick = 1e-4
        if bat is not None:
            mgr.bats = [None, bat]
            bat.show()
        else:
            continue
        mgr.start()
        dx = mgr.ball.TableBouncePoint[1] - target
        li.append(dx)
        sum += math.fabs(dx)
        if len(li) == 25:
            break
    print("average error:", sum / len(li))
