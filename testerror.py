from PingPong import *
import pickle

if __name__ == '__main__':
    robot = ServeBallRobot("data.txt")
    target = -1
    sum = 0
    li = []
    tryTimes = 0
    while True:
        try:
            tryTimes += 1
            ball = robot.GenerateBall()
            bat = CatchBall_Bat(ball, target)
            mgr = PingPongManager(ball)
            mgr.tick = 1e-3
            if bat is not None:
                mgr.bats = [None, bat]
            else:
                continue
            mgr.start()
            dx = mgr.ball.TableBouncePoint[1] - target
            li.append(dx)
            sum += math.fabs(dx)
            if len(li) == 300:
                break
        except Exception as e:
            print(f"{ball.vx}")
    print("average error:", sum / len(li))
    print("success rate:", len(li) / tryTimes)
