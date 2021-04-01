import math
import pickle

import numpy as np
import pygame
import torch

from PingPong3d import PingPongMgr3,DroppingStatus
from ServeBall3d import ServeBallRobot,ServeBallBatRobot, convertData2Bat, convertData2Ball

# from LearnHitBall import net, netE2,netE1
from LearnHitBall2 import net

width = 1080
height = 720
scale = 1000
rate = 0.25

modelfile = 'model/model-j1.bin'
throwfile = 'data/throw-5w-test.txt'
datafile = 'data/catch_more30'
dataflag = False
show_iter = 20

net_color = (238, 221, 130)
table_color = (92, 172, 238)
bat_color = (255, 69, 0)
ball_color = (255, 165, 0)
shadow_color= (145, 145, 145)

light_pos = np.array([0, 0, 10])
ob_center = np.array([2, 0, 1.4])
ob_theta = math.pi / 4
ob_fov = math.pi / 3


def rotate(pos: list, center: list, theta, phi):
    pos = np.array(pos)
    center = np.array(center)
    vec = pos - center
    A = np.array([
        [np.cos(theta), np.cos(np.pi / 2 + theta), 0],
        [np.sin(theta), np.sin(np.pi / 2 + theta), 0],
        [0, 0, 1]
    ])
    B = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])

    trans = A @ (B @ vec)
    return (center + trans).tolist()


def IntersectionWithSurface(p1, p2, flat):
    e = np.array(flat[0:3])
    dp = p2 - p1
    h = math.fabs(p1.dot(e) + flat[3])
    d = math.fabs(dp.dot(e))
    return p1 + dp * h / d


def convert3to2(p, center, e1, e2):
    d = p - center
    dx = d.dot(e1) / np.linalg.norm(e1)
    dy = d.dot(e2) / np.linalg.norm(e2)
    return [dy * scale + width / 2, dx * scale + height / 2]


if __name__ == '__main__':

    def testCartesian():
        p = [
            [0, 0, 0],
            [0.7625, 0, 0],
            [0, 1.37, 0],
            [0, 0, 1]
        ]
        p_project = []
        for i in p:
            tmp = IntersectionWithSurface(np.array(i), ob_pos, ob_surface)
            p_project.append(convert3to2(tmp, ob_center, ob_ex, ob_ey))

        pygame.draw.line(surface, (255, 0, 0), p_project[0], p_project[1], 3)
        pygame.draw.line(surface, (0, 255, 0), p_project[0], p_project[2], 3)
        pygame.draw.line(surface, (0, 0, 255), p_project[0], p_project[3], 3)


    def transPolyn(points):
        result = []
        for p in points:
            p_trans = IntersectionWithSurface(ob_pos, np.array(p), ob_surface)
            result.append(convert3to2(p_trans, ob_pos, ob_ex, ob_ey))
        return result


    def drawBall(center: list, radius):
        # 这里做一个近似
        center = np.array(center)
        project_point = IntersectionWithSurface(ob_pos, center, ob_surface)
        show_point = convert3to2(project_point, ob_center, ob_ex, ob_ey)
        eta = np.linalg.norm(project_point - ob_pos) / np.linalg.norm(center - ob_pos)
        radius = radius * eta * scale
        pygame.draw.circle(surface, ball_color, show_point, radius)


    def drawTable(table: list):
        table_trans = transPolyn(table)
        pygame.draw.polygon(surface, table_color, table_trans)


    def drawNet(net: list):
        net_trans = transPolyn(net)
        pygame.draw.polygon(surface, net_color, net_trans, 2)


    def drawBat(bat):
        center = [bat.x, bat.y, bat.z]
        th = bat.theta
        phi = bat.phi
        h = bat.h / 2
        w = bat.w / 2
        corners = [
            [center[0] - h, center[1] - w, center[2]],
            [center[0] - h, center[1] + w, center[2]],
            [center[0] + h, center[1] + w, center[2]],
            [center[0] + h, center[1] - w, center[2]]
        ]
        r_corner = [rotate(corner, center, th, phi) for corner in corners]
        trans_corner = transPolyn(r_corner)
        pygame.draw.polygon(surface, bat_color, trans_corner)

    def drawShadow(center:list,radius):
        # 这里做一个近似
        p_ball = np.array(center)
        p_shadow = IntersectionWithSurface(light_pos,p_ball,[0,0,1,0])
        if np.fabs(p_shadow[0])<=0.7625+radius and np.fabs(p_shadow[1])<1.37+radius:
            r_shadow = np.linalg.norm(light_pos-p_shadow)/np.linalg.norm(light_pos-p_ball) *radius
            p_show = IntersectionWithSurface(p_shadow,ob_pos,ob_surface)
            r = np.linalg.norm(ob_pos-p_show)/np.linalg.norm(ob_pos-p_shadow)*radius*scale
            p_show = convert3to2(p_show,ob_center,ob_ex,ob_ey)
            pygame.draw.circle(surface,shadow_color,p_show,r)



    pygame.init()
    clock = pygame.time.Clock()
    tps = 60

    ob_d = height / 2 / math.tan(ob_fov / 2) / scale
    ob_pos = ob_center.tolist()
    ob_pos[0] = ob_pos[0] + math.cos(ob_theta) * ob_d
    ob_pos[2] = ob_pos[2] + math.sin(ob_theta) * ob_d
    ob_pos = np.array(ob_pos)
    ob_e = np.array([math.cos(ob_theta), 0, math.sin(ob_theta)])
    ob_surface = np.concatenate((ob_e, np.array([-ob_e.dot(ob_center)])))
    ob_ex = np.array([math.cos(ob_theta - math.pi / 2), 0, math.sin(ob_theta - math.pi / 2)])
    ob_ey = np.array([0, 1, 0])

    surface = pygame.display.set_mode((width, height))

    # prepare table and net
    pygame.display.set_caption("3d-pingpongmaster")
    table = [
        [0.7625, 1.37, 0],
        [-0.7625, 1.37, 0],
        [-0.7625, -1.37, 0],
        [0.7625, -1.37, 0]
    ]
    net_corner = [
        [0.7625, 0, 0],
        [0.7625, 0, 0.1525],
        [-0.7625, 0, 0.1525],
        [-0.7625, 0, 0]
    ]

    with open(modelfile, 'rb') as f:
        n = pickle.load(f)

    if dataflag:
        bot = ServeBallBatRobot(datafile)
    else:
        bot = ServeBallRobot(throwfile)
    mgr = PingPongMgr3(None)

    iter = 0
    b = False

    while True:
        surface.fill((255, 250, 250))

        # plot table and net
        drawTable(table)
        drawNet(net_corner)
        if not b:
            iter += 1
            if dataflag:
                b,bat,_ = bot.generate(0.1)
            else:
                data = bot.generateNumber(np.random.randint(len(bot.lines)))
                b = convertData2Ball(data, 0.1)

                x_drop, y_drop, z_drop, v = DroppingStatus(b)
                b_data = torch.cat((torch.tensor(data), torch.tensor([x_drop, y_drop, z_drop, v.x, v.y, v.z])), 0)
                bat_data = n(b_data)
                #
                # bat_data = n(b)

                bat = convertData2Bat(bat_data.detach().numpy())
            mgr = PingPongMgr3(b)
            mgr.bats[1] = bat
        if iter > show_iter:
            break

        drawBat(mgr.bats[1])
        ball_center = [mgr.ball.x,mgr.ball.y,mgr.ball.z]
        drawShadow(ball_center,mgr.ball.radius)
        drawBall(ball_center, mgr.ball.radius)
        for i in range(int(1 / tps / mgr.tick * rate)):
            b = mgr.move()
            if not b:
                break

        pygame.display.update()
        clock.tick(tps)
