# -*- coding: utf-8 -*-

from random import random, choice, shuffle, sample, uniform
import os
import math
from genetic import Genetic
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from enum import IntEnum
from mpl_toolkits.mplot3d import Axes3D
from axiacore_parser import Parser

fig ,ax = plt.subplots(figsize=(1, 1), facecolor='w')
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.set_zlabel('F(x, y)')

font = {'family' : 'Calibri',
		'size'   : 16}

matplotlib.rc('font', **font)

points = [(0.1,5), (2,0.3)]

to_plot = [points]
to_plot.append(Genetic.simple_crossover(points[0], points[1]))
to_plot.append(Genetic.arithmetical_crossover(points[0], points[1], 0.3))
to_plot.append(Genetic.geometrical_crossover(points[0], points[1], 0.3))
to_plot.append(Genetic.linear_crossover(points[0], points[1]))
to_plot.append(Genetic.BLX_crossover(points[0], points[1], 0.25))

colors = iter(cm.rainbow(np.linspace(0, 1, len(to_plot))))

for i, population in enumerate(to_plot):
	xs = [point[0] for point in population]
	ys = [point[1] for point in population]
	zs = 1

	color = next(colors)

	if i==0:
		label = u'до кроссовера'
	elif i==1:
		label = u'простой'
	elif i==2:
		label = u'арифметический'
	elif i==3:
		label = u'геометрический'
	elif i==4:
		label = u'линейный'
	elif i==5:
		label = u'смешанный'

	ax.plot(xs, ys, 'o', c=color, label=label, markersize=15)

plt.legend(loc=9, ncol=2, numpoints=1, fontsize=20)		
plt.margins(0.05)
plt.show()

