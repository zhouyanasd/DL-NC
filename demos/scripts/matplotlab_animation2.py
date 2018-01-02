# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#第一个参数必须为framenum
def update_line(num, data, line):
  line.set_data(data[...,:num])
  return line,
fig1 = plt.figure()
data = np.random.rand(2, 15)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')
#framenum从1增加大25后,返回再次从1增加到25,再返回...
line_ani = animation.FuncAnimation(fig1, update_line, 25,fargs=(data, l),interval=50, blit=True)
#等同于
#line_ani = animation.FuncAnimation(fig1, update_line, frames=25,fargs=(data, l),
#  interval=50, blit=True)
#忽略frames参数,framenum会从1一直增加下去知道无穷
#由于frame达到25以后,数据不再改变,所以你会发现到达25以后图形不再变化了
#line_ani = animation.FuncAnimation(fig1, update_line, fargs=(data, l),
#  interval=50, blit=True)
plt.show()