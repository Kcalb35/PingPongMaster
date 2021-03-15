import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def reflection(v, theta, phi, decay=0):
    e = polar2xyz(theta, phi)
    v_vertical = v.project(e)
    return v - v_vertical.multiply(1+ math.sqrt(1-decay))


def polar2xyz(theta, phi):
    z = math.cos(phi)
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    return vector3(x, y, z)


def Point2SurfaceDistance(x, y, z, x0, y0, z0, theta, phi):
    e = polar2xyz(theta, phi)
    p = vector3(x0, y0, z0)
    A = e.x
    B = e.y
    C = e.z
    D = -p.dot(e)
    distance = math.fabs(A * x + B * y + C * z + D) / math.sqrt(A ** 2 + B ** 2 + C ** 2)
    return distance


class vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.length = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        self.decay = 0.1

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
    def __init__(self, x, y, z, vx, vy, vz, decay=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.v = vector3(vx, vy, vz)
        self.radius = 0.02
        self.trace = [[x], [y], [z]]
        self.TableBouncePoint = []
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
        if d <= self.radius:
            x, y, z = self.TryMove(1e-5)
            d_next = Point2SurfaceDistance(x, y, z, target.x, target.y, target.z, target.theta, target.phi)
            if d_next < d:
                return True
        return False

    def TableBounce(self):
        self.v.z = -self.v.z * math.sqrt(1 - self.decay)
        self.TableBouncePoint.append(self.y)

    def Bounce(self, bat):
        v_after = reflection(self.v, bat.theta, bat.phi)
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
        ax.plot_surface(x, y, x * 0, color='b', alpha=0.5)

        x = np.linspace(0.7625, -0.7625, 4)
        z = np.linspace(0, 0.1525, 4)
        x, z = np.meshgrid(x, z)
        ax.plot_surface(x, x * 0, z, color='y', alpha=0.4)
        plt.ylim((-1.5, 1.5))
        plt.xlim((-1.5, 1.5))


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    b = PingPongBall3(0, 0, 0.2, 0.5, 5, 0.5, 0.3)
    bat1 = PingPongBat3(0, 1.37, 0, 0, 0, 0, math.pi / 2, math.pi / 2)
    mgr = PingPongMgr3(b)
    mgr.bats[1] = bat1
    mgr.start()
    mgr.show(ax)

    plt.show()
