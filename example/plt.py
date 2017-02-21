# -----------------------------------------------------------------------------
# Copyright (c) 2015, Nicolas P. Rougier. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(8,5), dpi=80)
# plt.subplot(111)
#
# X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
# C,S = np.cos(X), np.sin(X)
#
# plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
# plt.plot(X, S, color="red", linewidth=2.5, linestyle="-")
#
# plt.xlim(-4.0,4.0)
# plt.xticks(np.linspace(-4,4,9,endpoint=True))
#
# plt.ylim(-1.0,1.0)
# plt.yticks(np.linspace(-1,1,5,endpoint=True))
#
# plt.show()

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000)
y = np.sin(x)
z = np.cos(x**2)

plt.figure(figsize=(8,4))
plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
plt.plot(x,z,"b--",label="$cos(x^2)$")
plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("PyPlot First Example")
plt.ylim(-1.2,1.2)
plt.legend()
plt.show()
