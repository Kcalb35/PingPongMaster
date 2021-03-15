import math
import random
import matplotlib.pyplot as plt


def Point2LineDistance(x, y, line_x, line_y, theta):
    return math.fabs(math.sin(theta) * (x - line_x) - math.cos(theta) * (y - line_y))


def Point2PointDistance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def DroppingPoint(ball):
    a = ball.vx * ball.vy
    b = ball.vx * math.sqrt(ball.vy ** 2 + 2 * 9.8 * (ball.y - ball.radius))
    return (a + b) / 9.8 + ball.x, (a - b) / 9.8 + ball.x


def InterceptPoint(ball, catch_x):
    drop, _ = DroppingPoint(ball)
    a = 2 * drop - ball.x
    return catch_x, ball.y + ball.vy / ball.vx * (a - catch_x) - 9.8 / 2 / ball.vx ** 2 * (a - catch_x) ** 2


def SolveThetaEquation(a, b, c):
    delta = a ** 2 * b ** 2 + b ** 4 - b ** 2 * c ** 2
    # 无解
    if delta < 0:
        return False, 0, 0
    root = math.sqrt(delta)
    x1 = -(root + a * c) / (a ** 2 + b ** 2)
    y1 = (-a * x1 - c) / b
    x2 = (root - a * c) / (a ** 2 + b ** 2)
    y2 = (-a * x2 - c) / b

    return True, math.atan2(y1, x1) / 4 + math.pi / 2, math.atan2(y2, x2) / 4 + math.pi / 2


def Generate_bat(catch_x, catch_y, vx, vy, theta):
    # 留出碰撞位置
    catch_x += 0.02 / math.fabs(math.sin(theta))
    bat = PingPongBat(catch_x, catch_y, vx, vy, theta)

    # 球板触碰桌子则调整位置
    lower = min(bat.y - math.sin(bat.theta) * bat.radius, bat.y - math.sin(bat.theta) * bat.radius)
    if lower <= 0:
        lower = math.fabs(lower)
        bat.x += lower / math.sin(bat.theta) * math.cos(bat.theta)
        bat.y += lower
    return bat


def CatchBall_Bat(ball, target, dx=0.1):
    startflag = True

    # 计算一个合理的反弹点
    drop, _ = DroppingPoint(ball)
    catch_x, catch_y = InterceptPoint(ball, drop + dx)
    while catch_y < 0.02:
        dx -= 0.01
        catch_x, catch_y = InterceptPoint(ball, drop + dx)

    # 碰撞前的速度
    vx1 = ball.vx
    vy1 = -ball.vy + 9.8 / ball.vx * (2 * drop - ball.x - catch_x)

    # 初猜的数据
    iterCount = 100
    theta = math.pi / 4
    bat_v = vector(-0.5, 0.5)
    ball_v = vector(vx1, vy1)
    i = 0

    while i <= iterCount:
        catch_x, catch_y = InterceptPoint(ball, drop + dx)
        # 碰撞前的速度
        vx1 = ball.vx
        vy1 = -ball.vy + 9.8 / ball.vx * (2 * drop - ball.x - catch_x)
        ball_v = vector(vx1, vy1)
        i += 1
        if vy1 <= 0:
            startflag = False
        if startflag:
            a = (catch_y - ball.radius) * (vx1 ** 2 - vy1 ** 2) - 2 * vx1 * vy1 * (target - catch_x)
            b = 2 * (catch_y - ball.radius) * vx1 * vy1 + (target - catch_x) * (vx1 ** 2 - vy1 ** 2)
            c = (catch_y - ball.radius) * (vx1 ** 2 + vy1 ** 2) - 9.8 * (target - catch_x) ** 2
            thetas = SolveThetaEquation(a, b, c)
            # 有解
            if thetas[0]:
                theta = thetas[2]
                # 触网判断，如果触网向后调整击球点
                vx_bounce, vy_bounce = flection(vx1, vy1, 0, 0, theta)
                ball_bounce = PingPongBall(catch_x, catch_y, vx_bounce, vy_bounce)
                if NetTouchCheck(ball_bounce):
                    dx += 0.05
                    print("静拍有解触网")
                    continue
                # 不触网，则用低回
                return Generate_bat(catch_x, catch_y, 0, 0, theta)
            # 无解
            else:
                startflag = False
        else:
            equiv_v = ball_v - bat_v.multiply(2) + bat_v.project(theta).multiply(2)
            # 解方程
            square_sub = equiv_v.x ** 2 - equiv_v.y ** 2
            a = (catch_y - ball.radius) * square_sub - 2 * equiv_v.x * equiv_v.y * (target - catch_x)
            b = 2 * (catch_y - ball.radius) * equiv_v.x * equiv_v.y + (target - catch_x) * square_sub
            c = (catch_y - ball.radius) * (equiv_v.x ** 2 + equiv_v.y ** 2) - 9.8 * (target - catch_x) ** 2
            thetas = SolveThetaEquation(a, b, c)
            # 无解
            if not thetas[0]:
                print(f"加速:{bat_v.x:.1f},{bat_v.y:.1f} 击球点{catch_x:.2f},{catch_y:.2f} 无解")
                if catch_x <= 0.5:
                    bat_v.y += 0.1
                    bat_v.x -= 0.1
                else:
                    bat_v.x -= 0.2
                continue
            # 有解且收敛
            elif math.fabs(thetas[2] - theta) <= 1e-6:
                theta = thetas[2]
                # 检查触网
                vx_bounce, vy_bounce = flection(vx1, vy1, bat_v.x, bat_v.y, theta)
                ball_bounce = PingPongBall(catch_x, catch_y, vx_bounce, vy_bounce)
                if NetTouchCheck(ball_bounce):
                    print("触网")
                    bat_v.y += 0.1
                    continue

                print(f"加速 {bat_v.x:.2f},{bat_v.y:.2f} 角度 {theta} 有解")
                return Generate_bat(catch_x, catch_y, bat_v.x, bat_v.y, theta)
            # 有解未收敛
            else:
                theta = thetas[2]
    # 无解
    return None


def NetTouchCheck(ball):
    y_net = ball.y + ball.vy / ball.vx * -ball.x - 9.8 / 2 / ball.vx ** 2 * ball.x ** 2
    if y_net <= ball.radius + 0.1525:
        return True
    else:
        return False


class vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = math.sqrt(self.x ** 2 + self.y ** 2)

    def __sub__(self, other):
        return vector(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return vector(self.x + other.x, self.y + other.y)

    def __neg__(self):
        return vector(-self.x, -self.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def project(self, theta):
        h = self.x * math.cos(theta) + self.y * math.sin(theta)
        return vector(h * math.cos(theta), h * math.sin(theta))

    def multiply(self, times):
        return vector(self.x * times, self.y * times)


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
        self.TableBouncePoint = []

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
        d = Point2LineDistance(self.x, self.y, targetBat.x, targetBat.y, targetBat.theta)
        # 这里用中心距离判断有点小问题
        dr = Point2PointDistance(self.x, self.y, targetBat.x, targetBat.y)

        if d <= self.radius and dr <= math.sqrt(targetBat.radius ** 2 + self.radius ** 2):
            x, y = self.TryMove(1e-5)
            d_next = Point2LineDistance(x, y, targetBat.x, targetBat.y, targetBat.theta)
            if d_next < d:
                return True
        return False

    def Bounce(self, targetBat):
        self.vx, self.vy = flection(self.vx, self.vy, targetBat.vx, targetBat.vy, targetBat.theta)

    def CheckTableBounce(self):
        if self.y <= self.radius and math.fabs(self.x) <= 1.37:
            return True
        else:
            return False

    def TableBounce(self):
        self.vy = -self.vy
        self.TableBouncePoint.append(self.x)

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
        plt.plot([self.x - dx, self.x + dx], [self.y - dy, self.y + dy], color='r')


class ServeBallRobot:
    def __init__(self, file):
        with open(file, "r") as f:
            self.lines = f.readlines()

    def GenerateBall(self):
        item = random.randint(0, len(self.lines) - 1)
        print(item)
        return self.GenerateBallbyIndex(item)

    def GenerateBallbyIndex(self, index):
        item = self.lines[index].replace('\n', '').split(' ')
        splited = [float(m) for m in item if m != '']
        b = PingPongBall(splited[0], splited[1], splited[2], splited[3])
        return b


class PingPongManager:
    def __init__(self, ball):
        self.ball = ball
        self.tick = 1e-4
        self.bats = [None, None]

    def start(self):
        while True:
            self.ball.Move(self.tick)
            if self.ball.CheckTableBounce():
                self.ball.TableBounce()
                if len(self.ball.TableBouncePoint) >= 2 and self.ball.TableBouncePoint[-1] * self.ball.TableBouncePoint[
                    -2] > 0:
                    break

            # if self.ball.CheckTouchNet() or self.ball.CheckOutBoundary():
            if self.ball.CheckOutBoundary():
                break
            for i in range(2):
                if self.bats[i] is not None and self.ball.CheckBounce(self.bats[i]):
                    self.ball.Bounce(self.bats[i])

    def show(self):
        # plt.figure(figsize=(24 / 2.54, 8 / 2.54),dpi=300)
        self.ball.showTrace()
        plt.plot([-1.37, 1.37], [0, 0])
        plt.plot([0, 0], [0, 0.1525])
        plt.ylim((-0.05, 1))
        # plt.xlim(-1.5, 1.5)
        # plt.show()


def flection(vx, vy, batvx, batvy, theta):
    ball_v = vector(vx, vy)
    ball_hv = ball_v.project(theta)
    ball_vv = ball_v - ball_hv

    bat_v = vector(batvx, batvy)
    bat_hv = bat_v.project(theta)
    bat_vv = bat_v - bat_hv

    # 进行速度叠加
    ball_vv = bat_vv.multiply(2) - ball_vv
    ball_v = ball_hv + ball_vv
    return ball_v.x, ball_v.y


if __name__ == '__main__':
    robot = ServeBallRobot("data.txt")
    target = -0.4
    # ball = robot.GenerateBall()
    ball = robot.GenerateBallbyIndex(970)
    print(ball.vx, ball.vy)
    bat = CatchBall_Bat(ball, target)
    mgr = PingPongManager(ball)
    if bat is not None:
        mgr.bats = [None, bat]
        bat.show()
    else:
        print("无解")
    mgr.start()
    mgr.show()
    if bat is not None:
        bat.show()
    plt.show()

    # for i in range(20):
    #     ball = ServeBallRobot().GenerateBallbyIndex(195)
    #     dropping = DroppingPoint(ball)[0] + 0.1
    #     x, y = InterceptPoint(ball, dropping)
    #     theta = math.pi / 21 * (i+1)
    #     catch_x = x + ball.radius / math.fabs(math.sin(theta))
    #     bat = PingPongBat(catch_x, y, 0, 0, theta)
    #     mgr = PingPongManager(ball)
    #     mgr.bats = [None, bat]
    #     mgr.start()
    #     mgr.show()
    #     bat.show()
    # plt.show()
