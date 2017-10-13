#!/usr/bin/env python3

import argparse
import numpy as np
from scipy.ndimage import gaussian_filter, imread
from scipy.ndimage.interpolation import shift


def litr(fname, show=False):
    if show:
        from matplotlib import pyplot as plt
    # получаем яркость картинки
    lightness = imread(fname)
    lightness = (lightness.min(axis=2) + lightness.max(axis=2)) / 2

    # сглаживаем
    lightness = gaussian_filter(lightness, 3)

    x = np.arange(lightness.shape[1])
    y = np.arange(lightness.shape[0])
    x, y = np.meshgrid(x, y)

    if show:
        plt.contourf(x, y, lightness)

    shifts = [(dx, dy) for dx in range(-1, 2)
                       for dy in range(-1, 2)
                       if dx != 0 or dy != 0]
    local_maximums = np.full(lightness.shape, True, dtype=bool)
    for s in shifts:
        local_maximums *= (lightness > shift(lightness, s))

    xm = x[local_maximums]
    ym = y[local_maximums]
    lm = lightness[local_maximums]

    # ищем 2 максимума
    i1 = lm.argmax()
    lm[i1] = 0
    i2 = lm.argmax()

    s1 = [xm[i1], ym[i1]]
    s2 = [xm[i2], ym[i2]]

    distance = ((s1[0] - s2[0]) ** 2 + (s1[1] - s2[1]) ** 2) ** .5

    if show:
        plt.plot([s1[0], s2[0]], [s1[1], s2[1]], color="k")
        plt.title("Distance: %d px" % round(distance))
        plt.show()

    return distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the distance between the two brightest points.')
    parser.add_argument('image', type=str, nargs=1,
                        help='path to image')
    parser.add_argument('-s', '--show', dest='show', action='store_true',
                        help='show colored contour plot')
    args = parser.parse_args()
    print(litr(args.image[0], args.show))