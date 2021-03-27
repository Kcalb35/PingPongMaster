import math
import matplotlib.pyplot as plt
import numpy as np


def reflection(v, batv, theta, phi, decay=0.1):
    """
    速度反射

    :param v:碰撞前的球速
    :param batv: bat velocity
    :param theta: bat theta
    :param phi: bat phi
    :param decay: energy decay
    :return: velocity after reflection
    """
    e = polar2xyz(theta, phi)
    v_vertical = v.project(e)
    batv_vertical = batv.project(e)
    eta = math.sqrt(1 - decay)
    return v - v_vertical.multiply(1 + eta) + batv_vertical.multiply(1 + eta)


def polar2xyz(theta, phi):
    """
    polar cartesian to xzy cartesian
    :param theta:
    :param phi:
    :return:
    """
    z = math.cos(phi)
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    return vector3(x, y, z)


def Point2SurfaceDistance(x, y, z, x0, y0, z0, theta, phi):
    """
    point to surface distance
    :param x:
    :param y:
    :param z:
    :param x0:
    :param y0:
    :param z0:
    :param theta:
    :param phi:
    :return:
    """
    e = polar2xyz(theta, phi)
    p = vector3(x0, y0, z0)
    A = e.x
    B = e.y
    C = e.z
    D = -p.dot(e)
    distance = math.fabs(A * x + B * y + C * z + D) / math.sqrt(A ** 2 + B ** 2 + C ** 2)
    return distance


def DroppingTime(ball):
    """
    calculate a ball from start drop to desk time
    :param ball:
    :return: dropping time
    """
    return (ball.v.z + math.sqrt(ball.v.z ** 2 + 2 * 9.8 * (ball.z - ball.radius))) / 9.8


def DroppingPoint(ball):
    """
    calculate dropping point
    :param ball:
    :return: x,y
    """
    t = DroppingTime(ball)
    return ball.x + t * ball.v.x, ball.y + t * ball.v.y


def DroppingStatus(ball):
    """
    getting dropping status
    :param ball:
    :return: x,y,z,v
    """
    x, y = DroppingPoint(ball)
    t = DroppingTime(ball)
    v = vector3(ball.v.x, ball.v.y, ball.v.z)
    v.z -= 9.8 * t
    v.z = -v.z * math.sqrt(1 - ball.decay)
    return x, y, 0.02, v


def InterceptPoint3(ball, d):
    """
    calculate intercept poing with ball and distance
    :param ball:
    :param d: shift distance
    :return: x,y,z, intercept point vz velocity
    """

    x, y, z, v_bounced = DroppingStatus(ball)
    vxy = math.sqrt(ball.v.x ** 2 + ball.v.y ** 2)
    dx = d * ball.v.x / vxy
    dy = d * ball.v.y / vxy
    catch_x = x + dx
    catch_y = y + dy

    dt = d / vxy
    catch_z = v_bounced.z * dt - 0.5 * 9.8 * dt ** 2 + ball.radius + z
    return catch_x, catch_y, catch_z, v_bounced.z - 9.8 * dt


def FastForward(ball, bat_v, catch_x, catch_y, catch_z, theta, phi):
    """
    从头快速计算碰撞。

    :param bat_v: bat velocity
    :param ball: ball with decay
    :param catch_x: bounce x
    :param catch_y: bounce y
    :param catch_z: bounce z
    :param theta: bat theta
    :param phi: bat phi
    :return: table bounce success, lat bounce point, last bounce point velocity
    """

    try:
        drop_x, drop_y, drop_z, v = DroppingStatus(ball)
    except Exception as e:
        return False, None, None, None

    up_t = math.sqrt((catch_x - drop_x) ** 2 + (catch_y - drop_y) ** 2) / math.sqrt(v.x ** 2 + v.y ** 2)
    v.z -= 9.8 * up_t
    v_after = reflection(v, bat_v, theta, phi, ball.decay)

    # check v_after y<0
    if v_after.y > 0:
        return False, None, None, None

    # check net pass
    dt = math.fabs(catch_y / v_after.y)
    h = catch_z + v_after.z * dt - 9.8 * 0.5 * dt ** 2
    # 0.1525 + 0.02 = 0.1725
    if h < 0.1725:
        return False, None, None, None

    # get status
    ball_bounced = PingPongBall3(catch_x, catch_y, catch_z, v_after.x, v_after.y, v_after.z, ball.decay)
    x_final, y_final, z_final, v_final = DroppingStatus(ball_bounced)

    # check out boundary
    if math.fabs(x_final) >= 0.7625 or y_final > -0.02 or y_final < -1.35:
        return False, None, None, None

    return True, (x_final, y_final, z_final), v, v_final


def correctBatPosition(v_ball, x, y, z, radius):
    """
    correct bat position
    :param v_ball:
    :param x:
    :param y:
    :param z:
    :param radius:
    :return: x,y,z
    """
    vec = v_ball.multiply(radius / v_ball.length)
    return x + vec.x, y + vec.y, z + vec.z


class vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.length = math.sqrt(x ** 2 + y ** 2 + z ** 2)

    def __sub__(self, other):
        return vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __neg__(self):
        return vector3(-self.x, -self.y, -self.z)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def multiply(self, times):
        return vector3(self.x * times, self.y * times, self.z * times)

    def project(self, other):
        l = self.dot(other) / other.length
        return other.multiply(l / other.length)


class PingPongBall3:
    def __init__(self, x, y, z, vx, vy, vz, decay=0.):
        self.x = x
        self.y = y
        self.z = z
        self.v = vector3(vx, vy, vz)
        self.radius = 0.02
        self.trace = [[x], [y], [z]]
        self.TableBouncePoint = [[], []]
        self.decay = decay

    def CheckOutBoundary(self):
        if (math.fabs(self.y) >= 1.37 or math.fabs(self.x) >= 0.7625) and self.z <= self.radius:
            return True
        else:
            return False

    def CheckTouchNet(self):
        if math.fabs(self.y) <= self.radius and math.fabs(self.x) <= 0.7625 and self.z <= 0.1525:
            return True
        else:
            return False

    def CheckTableBounce(self):
        if (math.fabs(self.y) <= 1.37 and math.fabs(self.x) <= 0.7625) and self.z <= self.radius:
            return True
        else:
            return False

    def CheckBounce(self, target):
        d = Point2SurfaceDistance(self.x, self.y, self.z, target.x, target.y, target.z, target.theta, target.phi)
        if d <= self.radius and math.sqrt(
                (self.x - target.x) ** 2 + (self.y - target.y) ** 2 + (self.z - target.z) ** 2) <= 0.075:
            x, y, z = self.TryMove(1e-5)
            d_next = Point2SurfaceDistance(x, y, z, target.x, target.y, target.z, target.theta, target.phi)
            if d_next < d:
                return True
        return False

    def TableBounce(self):
        self.v.z = -self.v.z * math.sqrt(1 - self.decay)
        self.TableBouncePoint[1].append(self.y)
        self.TableBouncePoint[0].append(self.x)

    def Bounce(self, bat):
        v_after = reflection(self.v, bat.v, bat.theta, bat.phi, self.decay)
        self.v = v_after

    def TryMove(self, time):
        x = self.x + time * self.v.x
        y = self.y + time * self.v.y
        z = self.z + self.v.z * time - 0.5 * 9.8 * time ** 2
        return x, y, z

    def Move(self, time):
        self.x, self.y, self.z = self.TryMove(time)
        self.v.z -= 9.8 * time
        self.trace[0].append(self.x)
        self.trace[1].append(self.y)
        self.trace[2].append(self.z)

    def showTrace(self):
        plt.plot(self.trace[0], self.trace[1], self.trace[2])


class PingPongBat3:
    def __init__(self, x, y, z, vx, vy, vz, theta, phi, w=0.156, h=0.150):
        self.x = x
        self.y = y
        self.z = z
        self.v = vector3(vx, vy, vz)
        self.theta = theta
        self.phi = phi
        self.w = w
        self.h = h


class PingPongMgr3:
    def __init__(self, ball):
        self.ball = ball
        self.tick = 1e-3
        self.bats = [None, None]

    def start(self):
        while True:
            self.ball.Move(self.tick)
            if self.ball.CheckTableBounce():
                self.ball.TableBounce()
                if len(self.ball.TableBouncePoint[0]) >= 2 and self.ball.TableBouncePoint[1][-1] * \
                        self.ball.TableBouncePoint[1][-2] > 0:
                    break
            if self.ball.CheckTouchNet() or self.ball.CheckOutBoundary():
                break
            for i in range(2):
                if self.bats[i] is not None and self.ball.CheckBounce(self.bats[i]):
                    self.ball.Bounce(self.bats[i])

    def drawBats(self):
        pass

    def show(self, ax):
        self.ball.showTrace()
        x = np.linspace(0.7625, - 0.7625, 4)
        y = np.linspace(1.37, -1.37, 4)
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, x * 0, color='b', alpha=0.1)

        x = np.linspace(0.7625, -0.7625, 4)
        z = np.linspace(0, 0.1525, 4)
        x, z = np.meshgrid(x, z)
        ax.plot_surface(x, x * 0, z, color='y', alpha=0.1)
        plt.ylim((-1.5, 1.5))
        plt.xlim((-1.5, 1.5))
