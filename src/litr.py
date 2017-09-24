#!/usr/bin/env python3

from scipy.misc import imread
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def greatest_neighbour(x, y, arr):
	neighbours = [[x, y, arr[x, y]]]
	if y > 0:
			neighbours.append([x, y-1, arr[x, y-1]])
	if y < arr.shape[1] - 1:
		neighbours.append([x, y+1, arr[x, y+1]])
	if x > 0:
		neighbours.append([x-1, y, arr[x-1, y]])
		if y > 0:
			neighbours.append([x-1, y-1, arr[x-1, y-1]])
		if y < arr.shape[1] - 1:
			neighbours.append([x-1, y+1, arr[x-1, y+1]])
	if x < arr.shape[0] - 1:
		neighbours.append([x+1, y, arr[x+1, y]])
		if y > 0:
			neighbours.append([x+1, y-1, arr[x+1, y-1]])
		if y < arr.shape[1] - 1:
			neighbours.append([x+1, y+1, arr[x+1, y+1]])
	m = max(neighbours, key = lambda x: x[2])
	if m[0] != x or m[1] != y:
		return m[0], m[1]
	return None

im = imread("../img/test3.jpg")
lightness = (im.min(axis=2) + im.max(axis=2)) / 2
del im

lightness = gaussian_filter(lightness, 3)
u, v = np.gradient(lightness)

x = np.arange(lightness.shape[0])
y = np.arange(lightness.shape[1])
x, y = np.meshgrid(y, x)
gr = np.sqrt(u * u + v * v)
gr /= gr.max()

plt.contourf(x, y, lightness)

results = set()

for i in range(1000):
	cx = np.random.choice(np.arange(lightness.shape[0]))
	cy = np.random.choice(np.arange(lightness.shape[1]))

	r = greatest_neighbour(cx, cy, lightness)
	while r:
		cx, cy = r
		r = greatest_neighbour(cx, cy, lightness)
	results.add((cx, cy, lightness[cx, cy]))

results = list(results)
results.sort(key=lambda x: -x[2])
s1, s2 = results[:2]
plt.plot([s1[1], s2[1]], [s1[0], s2[0]], color="k")
plt.title("Distance: %d px" % ((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2) ** .5)
# plt.streamplot(x, y, v, u, linewidth=lw, color='k')
plt.show()

