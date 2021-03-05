import math
import random
import matplotlib.pyplot as plt

'''calculate point to line distance'''


def PointToLineDistance(x, y, line_x, line_y, theta):
    return math.fabs(math.sin(theta) * (x - line_x) - math.cos(theta) * (y - line_y))


def Point2PointDistance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def DroppingPoint(ball):
    a = ball.vx * ball.vy
    b = ball.vx * math.sqrt(ball.vy ** 2 + 2 * 9.8 * (ball.y - ball.radius))
    return (a + b) / 9.8 + ball.x, (a - b) / 9.8 + ball.x


def InterceptPoint(ball, catch_x):
    catch_x -= ball.radius * ball.vx / math.sqrt(ball.vx ** 2 + ball.vy ** 2)
    drop, _ = DroppingPoint(ball)
    a = 2 * drop - ball.x
    return catch_x, ball.y + ball.vy / ball.vx * (a - catch_x) - 9.8 / 2 / ball.vx ** 2 * (a - catch_x) ** 2


def SolveThetaEquation(a, b, c):
    delta = a ** 2 * b ** 2 + b ** 4 - b ** 2 * c ** 2
    if delta < 0:
        raise Exception("无解")
    root = math.sqrt(delta)
    x1 = -(root + a * c) / (a ** 2 + b ** 2)
    y1 = (-a * x1 - c) / b
    x2 = (root - a * c) / (a ** 2 + b ** 2)
    y2 = (-a * x2 - c) / b

    return math.atan2(y1, x1) / 4 + math.pi / 2, math.atan2(y2, x2) / 4 + math.pi / 2


def CatchBall_Bat(ball, catch_x, target):
    catch_x, catch_y = InterceptPoint(ball, catch_x)
    drop, _ = DroppingPoint(ball)
    vx1 = ball.vx
    vy1 = -ball.vy + 9.8 / ball.vx * (2 * drop - ball.x - catch_x)
    a = (catch_y - ball.radius) * (vx1 ** 2 - vy1 ** 2) - 2 * vx1 * vy1 * (target - catch_x)
    b = 2 * (catch_y - ball.radius) * vx1 * vy1 + (target - catch_x) * (vx1 ** 2 - vy1 ** 2)
    c = -(catch_y - ball.radius) * (vx1 ** 2 - vy1 ** 2) - 9.8 * (target - catch_x) ** 2

    thetas = SolveThetaEquation(a, b, c)
    # 默认低回
    theta = thetas[1]

    # 触网判断
    vx_bounce, vy_bounce = flection(vx1, vy1, theta)
    ball_bounce = PingPongBall(catch_x, catch_y, vx_bounce, vy_bounce)
    flag = NetTouchCheck(ball_bounce)
    # 如果会触网，则用高回
    if flag:
        theta = thetas[0]

    # todo 提高精度
    # 球板触碰桌子则调整位置
    bat = PingPongBat(catch_x, catch_y, 0, 0, theta)
    lower = min(bat.y - math.sin(bat.theta) * bat.radius, bat.y - math.sin(bat.theta) * bat.radius)
    if lower <= 0:
        lower = math.fabs(lower)
        bat.x += lower / math.sin(bat.theta) * math.cos(bat.theta)
        bat.y += lower

    return bat


def NetTouchCheck(ball):
    y_net = ball.y + ball.vy / ball.vx * -ball.x - 9.8 / 2 / ball.vx ** 2 * ball.x ** 2
    if y_net <= ball.radius + 0.1525:
        return True
    else:
        return False


class PingPongBall:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = 0.02
        self.trace = [[x], [y]]
        self.decay = 1
        # self.decay = math.sqrt(0.9)

    def CheckOutBoundary(self):
        if math.fabs(self.x) > 1.37 and self.y <= self.radius:
            return True
        else:
            return False

    def CheckTouchNet(self):
        if math.fabs(self.x) < self.radius and self.y <= 0.1525:
            return True
        else:
            return False

    def CheckBounce(self, targetBat):
        d = PointToLineDistance(self.x, self.y, targetBat.x, targetBat.y, targetBat.theta)
        # 这里用中心距离判断有点小问题
        dr = Point2PointDistance(self.x, self.y, targetBat.x, targetBat.y)

        if d <= self.radius and dr <= math.sqrt(targetBat.radius ** 2 + self.radius ** 2):
            x, y = self.TryMove(1e-5)
            d_next = PointToLineDistance(x, y, targetBat.x, targetBat.y, targetBat.theta)
            if d_next < d:
                return True
        return False

    def Bounce(self, targetBat):
        self.vx, self.vy = flection(self.vx, self.vy, targetBat.theta)

    def CheckTableBounce(self):
        if self.y <= self.radius and math.fabs(self.x) <= 1.37:
            return True
        else:
            return False

    def TableBounce(self):
        self.vy = -self.vy

    def TryMove(self, time):
        x = self.x + self.vx * time
        y = self.y + self.vy * time - 0.5 * 9.8 * time ** 2
        return x, y

    def Move(self, time):
        self.x, self.y = self.TryMove(time)
        self.vy -= 9.8 * time
        self.trace[0].append(self.x)
        self.trace[1].append(self.y)

    def showTrace(self):
        plt.plot(self.trace[0], self.trace[1])


class PingPongBat:
    def __init__(self, x, y, vx, vy, theta, radius=0.074, rotation=0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.rotation = rotation
        # by default the width of a bat is 0.148m
        self.radius = radius

    def show(self):
        dx = math.cos(self.theta) * self.radius
        dy = math.sin(self.theta) * self.radius
        plt.plot([self.x - dx, self.x + dx], [self.y - dy, self.y + dy])


class ServeBallRobot:
    def __init__(self):
        with open("ServeBallData.txt", "r") as f:
            self.lines = f.readlines()

    def GenerateBall(self):
        item = random.randint(0, len(self.lines) - 1)
        return self.GenerateBallbyIndex(item)

    def GenerateBallbyIndex(self, index):
        item = self.lines[index].replace('\n', '').split(' ')
        splited = [float(m) for m in item if m != '']
        b = PingPongBall(splited[0], splited[1], splited[2], splited[3])
        return b


class PingPongManager:
    def __init__(self, ball):
        self.ball = ball
        self.tick = 1e-3
        self.bats = [None, None]

    def start(self):
        while True:
            self.ball.Move(self.tick)
            if self.ball.CheckTableBounce():
                self.ball.TableBounce()
                print(f"table bounce {self.ball.x}")
            if self.ball.CheckTouchNet() or self.ball.CheckOutBoundary():
                break
            for i in range(2):
                if self.bats[i] is not None and self.ball.CheckBounce(self.bats[i]):
                    self.ball.Bounce(self.bats[i])

    def show(self):
        self.ball.showTrace()
        plt.plot([-1.37, 1.37], [0, 0])
        plt.plot([0, 0], [0, 0.1525])
        # plt.ylim((0, 0.5))
        # plt.xlim(-1.5, 1.5)
        plt.show()


def flection(vx, vy, theta):
    theta = 2 * theta
    vx, vy = math.cos(theta) * vx + math.sin(theta) * vy, math.sin(theta) * vx - math.cos(theta) * vy
    return vx, vy


if __name__ == '__main__':
    robot = ServeBallRobot()
    ball = robot.GenerateBall()
    bat = CatchBall_Bat(ball, DroppingPoint(ball)[0] + 0.1, -0.9)
    bat.show()
    mgr = PingPongManager(ball)
    mgr.bats = [None, bat]
    mgr.start()
    mgr.show()
