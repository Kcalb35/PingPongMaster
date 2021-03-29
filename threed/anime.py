import pygame
import numpy as np
import math

width = 1080
height = 720
scale = 2000


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
    return dy * scale + width / 2, dx * scale + height / 2


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

        pygame.draw.polygon(surface, (30, 144, 255), table_trans)
        pygame.draw.line(surface, (255, 0, 0), p_project[0], p_project[1], 3)
        pygame.draw.line(surface, (0, 255, 0), p_project[0], p_project[2], 3)
        pygame.draw.line(surface, (0, 0, 255), p_project[0], p_project[3], 3)


    def transPolyn(points):
        result = []
        for p in points:
            p_trans = IntersectionWithSurface(ob_pos, np.array(p), ob_surface)
            result.append(convert3to2(p_trans, ob_pos, ob_ex, ob_ey))
        return result


    pygame.init()
    clock = pygame.time.Clock()
    tps = 60

    ob_center = np.array([1, 0, 1.25])
    ob_theta = math.pi / 4
    ob_phi = math.pi / 4
    ob_fov = math.pi / 2
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
    pygame.display.set_caption("3d-pingpongmaster")
    table = [
        [0.7625, 1.37, 0],
        [-0.7625, 1.37, 0],
        [-0.7625, -1.37, 0],
        [0.7625, -1.37, 0]
    ]
    table_trans = transPolyn(table)

    # prepare net
    net = [
        [0.7625, 0, 0],
        [0.7625, 0, 0.1525],
        [-0.7625, 0, 0.1525],
        [-0.7625, 0, 0]
    ]
    net_trans = transPolyn(net)
    while True:
        surface.fill((255, 250, 250))

        # plot table and net
        pygame.draw.polygon(surface, (30, 144, 250), table_trans)
        pygame.draw.polygon(surface, (165, 42, 42), net_trans,2)

        pygame.display.update()
        clock.tick(tps)
