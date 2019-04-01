import time
import numpy as np

class timelog():
    def __init__(self):
        self.itime = time.time()
        self.interation = 0
        with open('wall_time' + '.dat', 'w') as f:
            f.write('iteration' + ' '
                    + 'result' + ' '
                    + 'wall_time' + ' '
                    + '\n')

    @property
    def elapsed(self):
        return time.time() - self.itime

    def disp_time(self):
        temp = str(int(self.elapsed // 60)) + ':' + ("%2.1f" % (self.elapsed % 60)).rjust(4, '0')
        print(temp)
        return temp

    def save(self, result):
        self.interation += 1
        with open('wall_time' + '.dat', 'a') as f:
            f.write(str(self.interation) + ' ' + str(result) + ' ' + str(self.elapsed) + ' ' + '\n')


t = timelog()


def f(**p):
    result = (np.array(p['x'])) ** 2 + (np.array(p['y'])) ** 2 + 1
    time.sleep(1)
    t.save(result)
    return result