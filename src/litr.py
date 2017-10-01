#!/usr/bin/env python3

from scipy.misc import imread
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def greatest_neighbour(x, y, arr):
	# сдвиги по осям
	xshifts = np.arange(-1, 2)
	yshifts = np.arange(-1, 2)
	# индексы
	xs = x + xshifts
	ys = y + yshifts
	ys = ys[ys < arr.shape[0]]
	xs = xs[xs < arr.shape[1]]
	ys = ys[ys >= 0]
	xs = xs[xs >= 0]
	xs, ys = np.meshgrid(xs, ys)
	# положение максимума
	i = np.unravel_index(np.argmax(arr[ys, xs]), xs.shape)
	if xs[i] != x or ys[i] != y:
	 	return xs[i], ys[i]
	return None

if __name__ == '__main__':
	# получаем яркость картинки
	lightness = imread("../img/test1.jpg")
	lightness = (lightness.min(axis=2) + lightness.max(axis=2)) / 2

	# сглаживаем
	lightness = gaussian_filter(lightness, 3)

	x = np.arange(lightness.shape[0])
	y = np.arange(lightness.shape[1])
	x, y = np.meshgrid(x, y)
	
	plt.contourf(x, y, lightness)

	results = set()

	for i in range(1000):
		cx = np.random.choice(np.arange(lightness.shape[0]))
		cy = np.random.choice(np.arange(lightness.shape[1]))

		r = greatest_neighbour(cx, cy, lightness)
		while r:
			cx, cy = r
			r = greatest_neighbour(cx, cy, lightness)
		results.add((cx, cy, lightness[cy, cx]))

	results = list(results)
	results.sort(key=lambda x: -x[2])
	s1, s2 = results[:2]
	plt.plot([s1[0], s2[0]], [s1[1], s2[1]], color="k")
	plt.title("Distance: %d px" % ((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2) ** .5)
	plt.show()
