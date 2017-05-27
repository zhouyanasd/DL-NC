from brian2 import *

np.random.seed(100)

def Tri_function(duration):
    rng = np.random
    TIME_SCALE = defaultclock.dt
    in_number = int(duration/TIME_SCALE)

    def sin_fun(l, c, t):
        return (np.sin(c * t * TIME_SCALE/us) + 1) / 2

    def tent_map(l, c, t):
        temp = l
        if (temp < 0.5 and temp > 0):
            temp = (c / 101+1) * temp
            return temp
        elif (temp >= 0.5 and temp < 1):
            temp = (c / 101+1) * (1 - temp)
            return temp
        else:
            return 0.5

    def constant(l, c, t):
        return c / 100

    def chose_fun():
        c = rng.randint(0, 3)
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

    data = []
    cla = []
    cons = rng.randint(1, 101)
    fun, c = chose_fun()

    for t in range(in_number):
        if change_fun(0.7) and t % 50 ==0:
            cons = rng.randint(1, 101)
            fun, c = chose_fun()
            try:
                data_t= fun(data[t - 1], cons, t)
                data.append(data_t)
                cla.append(c)
            except IndexError:
                data_t = fun(rng.randint(1, 101)/100, cons, t)
                data.append(data_t)
                cla.append(c)
        else:
            try:
                data_t = fun(data[t - 1], cons, t)
                data.append(data_t)
                cla.append(c)
            except IndexError:
                data_t= fun(rng.randint(1, 101)/100, cons, t)
                data.append(data_t)
                cla.append(c)
    cla = np.asarray(cla)
    data = np.asarray(data)
    return data, cla

#----------------------------------------

duration = 200 * ms

data, cla = Tri_function(duration)

print('data: ', data.size)
print('cla: ', cla.size)


fig0 = plt.figure(figsize=(20, 4))
plot(data, 'r')
plot(cla,'.k')
show()