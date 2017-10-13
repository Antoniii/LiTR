#!/usr/bin/env python3

from scipy.ndimage import gaussian_filter, imread
from scipy.ndimage.interpolation import shift
from matplotlib import pyplot as plt
import numpy as np


def test(fname):
    # получаем яркость картинки
    lightness = imread(fname)
    lightness = (lightness.min(axis=2) + lightness.max(axis=2)) / 2

    # сглаживаем
    lightness = gaussian_filter(lightness, 3)

    x = np.arange(lightness.shape[1])
    y = np.arange(lightness.shape[0])
    x, y = np.meshgrid(x, y)

    plt.contourf(x, y, lightness)

    shifts = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]
    local_maximums = np.full(lightness.shape, True)
    for s in shifts:
        local_maximums *= (lightness > shift(lightness, s))
    print(local_maximums)

    # структурируем для сортировки
    results = np.core.records.fromarrays([x[local_maximums],
                                          y[local_maximums],
                                          lightness[local_maximums]],
                                         names='x, y, im',
                                         formats = 'i4, i4, f8')

    # сортировать ради двух наибольших -- оверкилл
    results.sort(order="im")

    s1 = results[-1]
    s2 = results[-2]
    plt.plot([s1[0], s2[0]], [s1[1], s2[1]], color="k")
    plt.title("Distance: %d px" % ((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2) ** .5)
    plt.show()

if __name__ == '__main__':
    test("../img/test1.jpg")
    test("../img/test2.jpg")
    test("../img/test3.jpg")