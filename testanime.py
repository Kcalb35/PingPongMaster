import pygame
from pygame.locals import *
from PingPong import *


class DisplayManager:
    def __init__(self, ball):
        self.tick = 1e-4
        self.ball = ball
        self.bats = [None, None]
        self.target = -0.7
        self.bats[1] = CatchBall_Bat(ball, self.GenerateTarget())
        self.bats[0] = PingPongBat(-1.4, 0.2, 0, 0, math.pi / 2)

    def GenerateTarget(self):
        return self.target - (random.random() - 0.5) * 0.3

    def Move(self, time):
        frames = time / self.tick
        for _ in range(int(frames)):
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
                    if i == 1:
                        tmpball = PingPongBall(-self.ball.x, self.ball.y, -self.ball.vx, self.ball.vy)
                        tmpbat = CatchBall_Bat(tmpball, self.GenerateTarget())
                        if tmpbat is None:
                            print("no solution for", tmpball.x, tmpball.y, tmpball.vx, tmpball.vy)
                        tmpbat.x = - tmpbat.x
                        tmpbat.vx = - tmpbat.vx
                        tmpbat.theta = math.pi - tmpbat.theta
                    else:
                        tmpbat = CatchBall_Bat(self.ball, self.GenerateTarget())
                    self.bats[1 - i] = tmpbat
                    break


if __name__ == '__main__':
    pygame.init()
    clock = pygame.time.Clock()
    scale = 300
    width = 1080
    height = 360
    tps = 60
    speed = 1

    windowSurface = pygame.display.set_mode((width, height))
    pygame.display.set_caption("pingpongMaster")

    robot = ServeBallRobot("data.txt")
    # mgr = DisplayManager(robot.GenerateBall())
    mgr = DisplayManager(robot.GenerateBallbyIndex(131))

    def convert(pos):
        return pos[0] * scale + width / 2, height - pos[1] * scale


    def drawBat(bat):
        dx = math.cos(bat.theta) * bat.radius
        dy = math.sin(bat.theta) * bat.radius

        pygame.draw.line(windowSurface, (223, 1, 1), convert((bat.x - dx, bat.y - dy + 0.02)),
                         convert((bat.x + dx, bat.y + dy + 0.02)), 3)


    while True:
        windowSurface.fill((255, 250, 250))

        # draw table and net
        pygame.draw.rect(windowSurface, (30, 144, 255),
                         ((width / 2 - 1.37 * scale, height - 0.02 * scale), (2.73 * scale, 0.02 * scale)))
        pygame.draw.line(windowSurface, (165, 42, 42), (width / 2, height - 0.02 * scale),
                         (width / 2, height - 0.1725 * scale), 2)

        mgr.Move(1 / tps)
        pygame.draw.circle(windowSurface, (255, 165, 0),
                           (mgr.ball.x * scale + width / 2, -mgr.ball.y * scale + height - 0.02 * scale),
                           mgr.ball.radius * scale)
        drawBat(mgr.bats[0])
        drawBat(mgr.bats[1])

        pygame.display.update()
        clock.tick(tps * speed)
