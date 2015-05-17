# -*- coding: utf-8 -*-

__author__ = 'smg'
from random import random, uniform

from enum import IntEnum


class Crossovers:
    def __init__(self):
        pass

    class Type(IntEnum):
        simple = 0
        arithmetical = 1
        geometrical = 2
        BLXalpha = 3
        linear = 4

    @staticmethod
    def cross(t1, t2, cross_type=Type.arithmetical, alpha=random()):
        return Crossovers.fncs[cross_type].__func__(t1, t2, alpha)

    @staticmethod
    def simple_crossover(t1, t2, *args):
        return (t1[0], t2[1]), (t2[0], t1[1])

    @staticmethod
    def arithmetical_crossover(t1, t2, a):
        ch1 = (t1[0] * a + t2[0] * (1 - a), t1[1] * a + t2[1] * (1 - a))
        ch2 = (t2[0] * a + t1[0] * (1 - a), t2[1] * a + t1[1] * (1 - a))
        return ch1, ch2

    @staticmethod
    def geometrical_crossover(t1, t2, a):
        ch1 = (t1[0] ** a * t2[0] ** (1 - a), t1[1] ** a * t2[1] ** (1 - a))
        ch2 = (t2[0] ** a * t1[0] ** (1 - a), t2[1] ** a * t1[1] ** (1 - a))
        return ch1, ch2

    @staticmethod
    def linear_crossover(t1, t2, *args):
        ch1 = (t1[0] * 0.5 + t2[0] * 0.5, t1[1] * 0.5 + t2[1] * 0.5)
        ch2 = (t1[0] * 1.5 - t2[0] * 0.5, t1[1] * 1.5 - t2[1] * 0.5)
        ch3 = (t2[0] * 1.5 - t1[0] * 0.5, t2[1] * 1.5 - t1[1] * 0.5)
        return ch1, ch2, ch3

    @staticmethod
    def blx_crossover(t1, t2, alpha):
        ch1 = (Crossovers.blx_gene(t1[0], t2[0], alpha), Crossovers.blx_gene(t1[1], t2[1], alpha))
        return [ch1]

    @staticmethod
    def blx_gene(g1, g2, alpha):
        cmin = min(g1, g2)
        cmax = max(g1, g2)
        length = cmax - cmin
        return uniform(cmin - length * alpha, cmax + length * alpha)

    fncs = {Type.simple: simple_crossover,
            Type.arithmetical: arithmetical_crossover,
            Type.geometrical: geometrical_crossover,
            Type.linear: linear_crossover,
            Type.BLXalpha: blx_crossover}
