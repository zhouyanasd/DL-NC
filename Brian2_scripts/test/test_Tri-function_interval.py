from brian2 import *

def Tri_function(duration, pattern_duration = 100, pattern_interval = 50, obj=-1):
    rng = np.random
    TIME_SCALE = defaultclock.dt
    in_number = int(duration / TIME_SCALE)
    data = []
    cla = []

    def sin_fun(l, c, t):
        return (np.sin(c * t * TIME_SCALE / us) + 1) / 2

    def tent_map(l, c, t):
        temp = l
        if (temp < 0.5 and temp > 0):
            temp = (c / 101 + 1) * temp
            return temp
        elif (temp >= 0.5 and temp < 1):
            temp = (c / 101 + 1) * (1 - temp)
            return temp
        else:
            return 0.5

    def constant(l, c, t):
        return c / 100

    def chose_fun():
        if obj == -1:
            c = rng.randint(0, 3)
        else:
            c = obj
        if c == 0:
            return sin_fun, c
        elif c == 1:
            return tent_map, c
        elif c == 2:
            return constant, c

    def change_fun(rate):
        fun = rng.randint(1, 101)
        if fun > 100 * rate:
            return False
        else:
            return True

    #---------------------------------
    for t in range(in_number):
        t_temp = t % pattern_duration
        if t_temp == 0:
            cons = rng.randint(1, 101)
            fun, c = chose_fun()
            cla.append(c)

        if t_temp < pattern_interval / 2:
            data.append(0)
        elif t_temp >= pattern_interval / 2 and t_temp < pattern_duration - pattern_interval / 2:
            data_t = fun(data[t - 1], cons, t)
            data.append(data_t)
        elif t_temp >= pattern_duration - pattern_interval / 2:
            data.append(0)
        else :
            data.append(0)
    return np.asarray(data), np.asarray(cla)

if __name__ == '__main__':
    data, label = Tri_function(100*ms)

    print(label)
    fig0 = plt.figure(figsize=(20, 4))
    subplot(211)
    plot(data, 'r')
    show()