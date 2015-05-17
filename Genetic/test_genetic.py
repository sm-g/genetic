import unittest
from genetic import Genetic
from math import fsum

__author__ = 'smg'


class TestGenetic(unittest.TestCase):
    def test_f(self):
        g = Genetic('x*y+y')
        res = g.f(1, 1)
        self.assertEqual(2, res)

    def test_normalize_fitness_max(self):
        res = Genetic.normalize_fitness([1, 1, 2], 'max')
        self.assertEqual([0, 0, 1], res)

    def test_normalize_fitness_sum_is_1(self):
        res = Genetic.normalize_fitness([1, 1, 2, 5, 1, 3], 'max')
        self.assertEqual(1, fsum(res))

    def test_normalize_fitness_min(self):
        res = Genetic.normalize_fitness([1, 2, 4, 1], 'min')
        self.assertEqual([0.375, 0.25, 0.0, 0.375], res)

    def test_normalize_fitness_equals_all_best(self):
        res = Genetic.normalize_fitness([1, 1, 1], 'max')
        self.assertEqual([1, 1, 1], res)

    def test_sorted_normed_fitness(self):
        res = Genetic.sorted_normed_fitness(lambda pop: [sum(i) for i in pop], 'max', [(1, 0), (3, 4)])
        self.assertEqual([(1, 1.0), (0, 0.0)], res)

    def test_sorted_normed_fitness2(self):
        res = Genetic.sorted_normed_fitness(lambda pop: [i for i in pop], 'min', [-1, 2, 0])
        self.assertEqual([(0,0.6), (2,0.4), (1,0.0)], res)

if __name__ == "__main__":
    unittest.main()
