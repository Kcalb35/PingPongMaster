from PingPong import *

if __name__ == '__main__':
    bot = ServeBallRobot('data.txt')
    count = 0
    for i in range(1000):
        if bot.GenerateBall().y>=0.3:
            count +=1
    print(count/1000)