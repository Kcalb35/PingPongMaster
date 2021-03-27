import random
import math
import sys
import matplotlib.pyplot as plt

from PingPong3d import PingPongBall3, PingPongBat3, DroppingPoint, PingPongMgr3, FastForward, InterceptPoint3, vector3, \
    correctBatPosition


def convertData2Ball(data, decay):
    return PingPongBall3(data[0], data[1], data[2], data[3], data[4], data[5], decay)


def convertData2Bat(data):
    return PingPongBat3(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])


def convertLine2Ball(line, decay):
    split = [float(m) for m in line.split(' ') if m != '']
    return convertData2Ball(split, decay)


class ServeBallRobot:
    def __init__(self, file):
        with open(file, 'r') as f:
            self.lines = f.readlines()
        self.length = len(self.lines)

    def generateBall(self, decay):
        return self.generateBallIndex(random.randint(0, self.length - 1), decay)

    def generateBallIndex(self, index, decay):
        splited = self.generateNumber(index)
        return convertData2Ball(splited, decay)

    def generateNumber(self, index):
        item = self.lines[index].replace('\n', '').split(' ')
        splited = [float(m) for m in item if m != '']
        return splited


class ServeBallBatRobot:
    def __init__(self, file):
        with open(file, 'r') as f:
            self.lines = f.readlines()
        self.length = len(self.lines)

    def generateBallBat(self, index, decay):
        ball_data, bat_data, score = self.generateNumbers(index)
        ball = convertData2Ball(ball_data, decay)
        bat = convertData2Bat(bat_data)
        return ball, bat, score

    def generateNumbers(self, index):
        item = self.lines[index].replace('\n', '').split(' ')
        splited = [float(m) for m in item if m != '']
        return splited[0:6], splited[6:14], splited[14]

    def generate(self, decay):
        return self.generateBallBat(random.randint(0, self.length - 1), decay)


def randomThrow(file, times, decay):
    lines = []
    while len(lines) < times:
        x = 0.7625 * random.uniform(-1, 1)
        y = -2 * random.random()
        z = 0.2 + 0.2 * random.random()

        vy = 8 * random.random()
        vx = 2 * random.uniform(-0.5, 0.5)
        vz = random.uniform(-0.5, 0.5)
        ball = PingPongBall3(x, y, z, vx, vy, vz, decay)

        dt = math.fabs(y / vy)
        h = z + vz * dt - 9.8 * 0.5 * dt ** 2
        if h <= 0.1525:
            continue

        x_drop, y_drop = DroppingPoint(ball)
        if math.fabs(x_drop) >= 0.7625 or y_drop > 1.35 or y_drop < 0.05:
            continue

        lines.append(f"{x} {y} {z} {vx} {vy} {vz}\n")
        print(f"{len(lines)}/{times}")
    with open(file, 'w') as f:
        f.writelines(lines)


def randomCatch(datafile, file, times, decay):
    with open(datafile, 'r') as f:
        lines = [line.replace('\n', '') for line in f.readlines()]

    results = []
    for i in range(0, times - 1):
        solution = []
        while len(solution) < 5:
            # prepare random bat data
            b = convertLine2Ball(lines[i], decay)
            x, y, z, _ = InterceptPoint3(b, random.uniform(0, 0.5))
            theta = random.random() * math.pi
            phi = random.random() * math.pi
            vx = random.uniform(-0.5, 0.5)
            vy = -random.random() / 2
            vz = random.uniform(-0.5, 0.5)

            flag, pos, v_mid, v_final = FastForward(b, vector3(vx, vy, vz), x, y, z, theta, phi)
            if flag:
                # calculate score
                score = math.fabs(pos[0] / 0.7625) + math.fabs(pos[1] / 1.37)
                x, y, z = correctBatPosition(v_mid, x, y, z, b.radius)
                solution.append([score, x, y, z, vx, vy, vz, theta, phi])
        solution.sort(key=lambda item: item[0], reverse=True)
        best = solution[0]
        ball = convertLine2Ball(lines[i], decay)
        bat = convertData2Bat(best[1:9])
        mgr = PingPongMgr3(ball)
        mgr.bats[1] = bat
        mgr.start()
        if len(mgr.ball.TableBouncePoint[1]) >= 2 and mgr.ball.TableBouncePoint[1][-1] < -0.02:
            line = [str(x) for x in best[1:9]]
            line.insert(0, lines[i])
            line.append(str(best[0]))
            results.append(' '.join(line) + '\n')
    with open(file, 'w') as f:
        f.writelines(results)


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        exit(0)

    elif args[1] == 'help':
        print('python ServeBall3d.py throw <data-file> <times> <decay>')
        print('python ServeBall3d.py catch <data-file> <save-file> <times> <decay>')
    elif args[1] == 'throw':
        param = args[2:]
        if len(param) != 3:
            print("Error: number of params")
            exit(233)
        file, times, decay = param
        randomThrow(file, int(times), float(decay))
    elif args[1] == 'catch':
        param = args[2:]
        if len(param) != 4:
            print("Error: number of params")
            exit(233)
        data, save, times, decay = param
        randomCatch(data, save, int(times), float(decay))
