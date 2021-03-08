import random
from PingPong import *

if __name__ == '__main__':
    tick = 1e-3
    mgr = PingPongManager(None)
    with open("data.txt", 'a') as f:
        for i in range(4000):
            # x :-2 ~ 0 , y:0~0.4
            x = -2 * random.random()
            y = 0.4 * random.random()
            # Vx:0~8, Vy:-1~1
            Vx = 8 * random.random()
            Vy = (random.random() - 0.5) * 2
            b = PingPongBall(x, y, Vx, Vy)
            mgr.ball = b
            mgr.start()
            if len(mgr.ball.TableBouncePoint)>=1 and mgr.ball.TableBouncePoint[0] >= 0.1:
                line = f"{x} {y} {Vx} {Vy}\n"
                print(line)
                f.write(line)
