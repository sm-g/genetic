import unittest
from genetic import Genetic, Sampling, print_with_score
from math import fsum

__author__ = 'smg'


class TestGenetic(unittest.TestCase):
    f = 'x*x+y*y'

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
        self.assertEqual([(0, 0.6), (2, 0.4), (1, 0.0)], res)

    def test_mutate(self):
        point, mutations = Genetic.mutate((1, 0), 1, 0.5, 0.5)
        self.assertNotEqual((1, 0), point)
        self.assertEqual(2, mutations)

    def test_mutate2(self):
        point, mutations = Genetic.mutate((1, 0), 0, 0.5, 0.5)
        self.assertEqual((1, 0), point)
        self.assertEqual(0, mutations)

    def test_select_best(self):
        res = Genetic.select_best(lambda pop: [i for i in pop], 'min', [-1, 2, 0], 2)
        self.assertEqual([-1, 0], res)

    def test_select_best_zero(self):
        res = Genetic.select_best(lambda pop: [i for i in pop], 'min', [-1, 2, 0], 0)
        self.assertEqual([], res)

    def test_select_best_too_much(self):
        pop = [-1, 2, 0]
        res = Genetic.select_best(lambda pop: [i for i in pop], 'min', pop, 4)
        self.assertEqual(pop, res)

    def test_stohastic(self):
        g = Genetic(TestGenetic.f, size=11, cp=1)
        population = [(0.2, 0.5), (-0.2, 1), (0.2, 0.2), (-1, 0.3), (0.4, 0.8)]
        fit = g.fit_population(population)
        g.start()
        print population
        print_with_score(population, g.fit_population(population))
        print 'avg fitness = ' + str(sum(fit) / float(len(fit)))
        print Sampling.stochastic_sampling(g.fit_population, g.extremum, population)


if __name__ == "__main__":
    unittest.main()
