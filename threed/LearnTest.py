import torch
import numpy
from tqdm import tqdm

from PingPong3d import PingPongMgr3
import pickle

from LearnHitBall import net
from ServeBall3d import ServeBallRobot, convertData2Ball, convertData2Bat

if __name__ == '__main__':
    modelfile = 'model/model-10w.bin'
    testfile = 'data/throw-10w-test.txt'
    decay = 0.1
    with open(modelfile, 'rb')as f:
        n = pickle.load(f)

    success = 0
    outleft = 0
    # 意味着没接住
    outright = 0
    other = 0
    bot = ServeBallRobot(testfile)
    num = len(bot.lines) - 1
    for i in tqdm(range(num)):
        ball = bot.generateNumber(i)
        predict = n(torch.tensor(ball))

        bat_predict = convertData2Bat(predict.detach().numpy())
        ball = convertData2Ball(ball, decay)

        mgr = PingPongMgr3(ball)
        mgr.bats[1] = bat_predict
        mgr.tick = 1e-3
        mgr.start()
        if len(mgr.ball.TableBouncePoint[1]) >= 2 and -1.37 < mgr.ball.TableBouncePoint[1][1] < 0:
            success += 1
        elif len(mgr.ball.TableBouncePoint[1]) == 1 and mgr.ball.v.y < 0:
            outleft += 1
        elif len(mgr.ball.TableBouncePoint[1]) == 1 and mgr.ball.v.y > 0 or len(mgr.ball.TableBouncePoint[1]) >= 2 and 0 < \
                mgr.ball.TableBouncePoint[1][1] < 1.37:
            outright += 1
        else:
            other += 1
    print("success rate:", success / num)
    print("out boundary:", outleft / num)
    print("no catch:", outright / num)
    print("other:", other / num)
