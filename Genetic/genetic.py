# -*- coding: utf-8 -*-

from random import random, shuffle, sample, uniform
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import axes3d, Axes3D
from enum import IntEnum

from axiacore_parser import Parser


class Crossovers:
    fncs = {}

    class Type(IntEnum):
        simple = 0
        arithmetical = 1
        geometrical = 2
        BLXalpha = 3
        linear = 4

    def __init__(self):
        Crossovers.fncs = {Crossovers.Type.simple: Crossovers.simple_crossover,
                           Crossovers.Type.arithmetical: Crossovers.arithmetical_crossover,
                           Crossovers.Type.geometrical: Crossovers.geometrical_crossover,
                           Crossovers.Type.linear: Crossovers.linear_crossover,
                           Crossovers.Type.BLXalpha: Crossovers.blx_crossover}

    @staticmethod
    def cross(t1, t2, type=Type.arithmetical, alpha=random()):
        return Crossovers.fncs[type](t1, t2, alpha)

    @staticmethod
    def simple_crossover(t1, t2):
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
    def linear_crossover(t1, t2):
        ch1 = (t1[0] * 0.5 + t2[0] * 0.5, t1[1] * 0.5 + t2[1] * 0.5)
        ch2 = (t1[0] * 1.5 - t2[0] * 0.5, t1[1] * 1.5 - t2[1] * 0.5)
        ch3 = (t2[0] * 1.5 - t1[0] * 0.5, t2[1] * 1.5 - t1[1] * 0.5)
        return ch1, ch2, ch3

    @staticmethod
    def blx_crossover(t1, t2, alpha):
        ch1 = (Genetic.blx_gene(t1[0], t2[0], alpha), Genetic.blx_gene(t1[1], t2[1], alpha))
        return [ch1]


class Genetic:
    class SamplingType(IntEnum):
        stochastic = 0
        remainderStochastic = 1
        rank = 2
        tournament = 3

    parser = Parser()

    def __init__(self, f_xy, extremum='max', mp=0.02, cp=0.1, size=30,
                 search_field=(-2, -2, 2, 2), sampling=SamplingType.stochastic, elite=0,
                 crossover=Crossovers.Type.simple, max_generations=1000, alpha=0.5):
        """
        :param f_xy:
        :param extremum: тип экстремума
        :param mp: вероятность мутации
        :param cp: вероятность кроссовера
        :param size: размер популяции
        :param search_field: поле поиска (minx, miny, maxx, maxy)
        :param sampling: тип отбора
        :param elite: кол-во лучших, переходящих в новое поколение
        :param crossover: тип размножения
        :param max_generations: максимальное количество поколений
        :param alpha: параметр для BLX кроссорвера
        """
        self.extremum = extremum
        self.mutation_p = mp
        self.crossover_p = cp
        self.size = size
        self.search_field = search_field
        self.max_generations = max_generations
        self.sampling = sampling
        self.elite = elite
        self.crossover_type = crossover
        self.alpha = alpha

        self.samplingDict = {
            Genetic.SamplingType.rank: self.rank_sampling,
            Genetic.SamplingType.tournament: self.tournament_sampling,
            Genetic.SamplingType.stochastic: self.stochastic_sampling,
            Genetic.SamplingType.remainderStochastic: self.remainder_stochastic_sampling}

        self.crossovers = 0
        self.mutations = 0
        self.func_xy = f_xy.lower()
        self.parsed_f = Genetic.parser.parse(self.func_xy)

        self.fig = plt.figure(0)
        self.ax = self.fig.add_axes([0, 0, 0.9, 1], projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('F(x, y)')

        self.same_best_counter = 0

    @staticmethod
    def generate(n, search_field, xmin=-0.5, ymin=-0.5, xsize=1, ysize=1):
        """Генерирует случайную популяцию

        :param n: количество особей в популяции
        :param search_field: поле поиска (minx, miny, maxx, maxy), используется для рассчета xmin, ymin, xsize, ysize
        :param xmin: минимальные значения для x и y
        :param ymin: минимальные значения для x и y
        :param xsize: размеры поля, в котором генерируются координаты x и y
        :param ysize: размеры поля, в котором генерируются координаты x и y
        """
        if search_field:
            xmin = search_field[0]
            ymin = search_field[1]
            xsize = search_field[2] - xmin
            ysize = search_field[3] - ymin
        points = [(random() * xsize + xmin, random() * ysize + ymin) for _ in xrange(n)]
        return points

    def mutate(self, t, p, *args):
        """Мутация особи.
        t - точка
        p - вероятность мутации
        *args - dx и dy
        Меняет координаты точки на случайное число в диапазоне [-dc; dc]
        """
        if len(args) > 0:
            dx = args[0]
        else:
            dx = (self.search_field[2] - self.search_field[0]) / 10

        if len(args) > 1:
            dy = args[1]
        else:
            dy = (self.search_field[3] - self.search_field[1]) / 10

        if random() < p:
            self.mutations += 1
            t = (t[0] + uniform(-dx, dx), t[1])
        if random() < p:
            self.mutations += 1
            t = (t[0], t[1] + uniform(-dy, dy))
        return t

    @staticmethod
    def blx_gene(g1, g2, alpha):
        cmin = min(g1, g2)
        cmax = max(g1, g2)
        length = cmax - cmin
        return uniform(cmin - length * alpha, cmax + length * alpha)

    def crossover(self, t1, t2):
        """Cкрещивание двух особей с некоторой вероятностью"""

        if random() < self.crossover_p:
            self.crossovers += 1
            alpha = self.alpha if self.crossover_type == Crossovers.Type.BLXalpha else random()
            return Crossovers.cross(t1, t2, self.crossover_type, alpha)

        return t1, t2

    def f(self, x, y):
        """Заданная функция, для которой ищется экстремум"""
        try:
            return self.parsed_f.evaluate({'x': x, 'y': y})
        except Exception as e:
            print 'ошибка при вычислении функции: ' + str(e)

    def fitness(self, population):
        """Приспособленность популяции.
        Возвращает массив значений целевой функции для каждой особи.
        :type population: list
        :rtype : list
        """
        return [self.f(ind[0], ind[1]) for ind in population]

    def fitness_normed(self, population):
        """Нормированная приспособленность особей популяции"""
        fit = self.fitness(population)
        return Genetic.normalize_fitness(fit, self.extremum)

    def sorted_normed_fitness(self, population):
        """Возвращает пары (номер особи в популяции, норма приспособленности),
        отсортированные по норме.
        """
        fit = self.fitness_normed(population)
        numbered_fit = [(i, x) for i, x in enumerate(fit)]
        numbered_fit.sort(key=lambda t: t[1])
        numbered_fit.reverse()
        return numbered_fit

    @staticmethod
    def normalize_fitness(fitness, extremum):
        """Нормирует приспособленность от 0 до 1. Сумма приспособленностей особей популяции = 1.
        :type extremum: str
        :type fitness: list

        """

        if extremum == 'max':
            m = min(fitness)
            s = sum(fitness) - len(fitness) * m
            if s == 0:
                return [1] * len(fitness)
            return [(f - m) / s for f in fitness]
        else:
            m = max(fitness)
            s = len(fitness) * m - sum(fitness)
            if s == 0:
                return [1] * len(fitness)
            return [(m - f) / s for f in fitness]

    @staticmethod
    def print_with_score(population, score):
        print '\n'.join(['{}:\t{: .3f} {: .3f}\t{: .3f}'.format(i, t[0][0], t[0][1], t[1]) for i, t in
                         enumerate(zip(population, score))])

    def rate_population(self, population, verbose=True):
        """Оценка качества популяции.
        Возвращает особь с максимальной приспособленностью.
        """
        fit = self.fitness(population)
        score = Genetic.normalize_fitness(fit, self.extremum)
        max_score = max(score)
        i = score.index(max_score)
        if verbose:
            Genetic.print_with_score(population, score)
            print u"{:-^30}\n" \
                  u"{}:\t{: .3f} {: .3f}\t{:.3f}\n" \
                  u"F(x,y) = {: .3f}".format(u' лучшая особь ', i,
                                             population[i][0], population[i][1],
                                             max_score, fit[i])
        return population[i][0], population[i][1]

    # отборы

    def remainder_stochastic_sampling(self, population):
        """Пропорциональный отбор
        Для каждой особи вычисляется отношение ее приспособленности к средней
        приспособленности популяции. Целая часть этого отношения указывает,
        сколько раз нужно записать особь в промежуточную популяцию,
        а дробная — это ее вероятность попасть туда еще раз.
        """
        score = self.fitness_normed(population)
        newpop = []
        for i, t in enumerate(population):
            r = score[i] * len(population)
            newpop.extend([t] * math.trunc(r))
            if random() < r - math.trunc(r):
                newpop.append(t)
        return newpop

    def stochastic_sampling(self, population):
        """Пропорциональный отбор 2
        Особи располагаются на колесе рулетки, так что размер сектора каждой особи
        пропорционален ее приспособленности. Изначально промежуточная популяция пуста.

        N раз запуская рулетку, выберем требуемое количество особей для записи
        в промежуточную популяцию. Ни одна выбранная особь не удаляется с рулетки.
        """
        score = self.fitness_normed(population)

        rights = []  # правые границы отрезков, заполняющих единичный отрезок рулетки
        for i in range(len(population)):
            rights.append(sum(score[:i + 1]))
        newpop = []
        for z in range(len(population)):
            r = random()
            for i in range(len(population)):
                if r < rights[i]:
                    newpop.append(population[i])
                    break

        return newpop

    def rank_sampling(self, population):
        """Ранковый отбор
        Для каждой особи вероятность попасть в промежуточную популяцию
        пропорциональна ее порядковому номеру в отсортированном списке
        приспособленности популяции.
        """

        rights = []  # правые границы отрезков, заполняющих единичный отрезок рулетки
        # для популяции из 3 особей длины отрезков: 3/6, 2/6, 1/6, границы: 1/6, 1/2, 1
        ss = 0
        size = len(population) * (len(population) + 1) / 2.0
        for i in range(len(population)):
            ss += (i + 1) / size
            rights.append(ss)

        newpop = []
        for t in self.sorted_normed_fitness(population):
            r = random()
            for i in range(len(population)):
                if r < rights[i]:
                    newpop.append(population[t[0]])
                    break
        return newpop

    def tournament_sampling(self, population):
        """Турнирный отбор
        Из популяции случайным образом выбирается t особей, и лучшая из них
        помещается в промежуточную популяцию. Этот процесс повторяется N раз,
        пока промежуточная популяция не будет заполнена.
        """
        newpop = []
        r = None
        for i in range(len(population)):
            while True:
                r = sample(population, 2)  # выбираем 2 разных особи
                if r[0] != r[1]:
                    break

            newpop.append(self.rate_population(r, verbose=False))

        return newpop

    def to_continue(self, generation, best, best_prev):
        """Условие продолжения поиска решения.
        generation - номер поколения
        best - лучшая особь в текущем поколении
        bestPrev - лучшая особь в предыдущем поколении
        """
        if generation == self.max_generations:
            print u"достигнут предел количества поколений"
            return False

        if best == best_prev:
            self.same_best_counter += 1

        if self.same_best_counter == 10:
            print u"приспособленность лучшей особи не меняется"
            return False

        return True

    def select_best(self, population, n):
        """Возвращает n лучших особей популяции
        """
        if n <= 0:
            return []
        return [population[x[0]] for x in self.sorted_normed_fitness(population)][:n]

    @staticmethod
    def convergence(population):
        """Условие вырождения популяции - все особи одинаковые"""
        return population.count(population[0]) == len(population)

    def evolve_step(self, population):
        """Шаг эволюции
        Отбор (селекция). Скрещивание (размножение). Мутация.
        """
        # элита
        elite = self.select_best(population, self.elite)

        # промежуточная популяция после отбора - те, кто будет размножаться
        population = self.samplingDict[self.sampling](population)

        if Genetic.convergence(population):
            raise Exception('convergence')

        # перемешиваем для случайного разбиения на пары для размножения
        shuffle(population)

        new_pop = []
        size = len(population)

        # кроссовер
        for i in range(0, size, 2):
            if i + 1 != size:
                t = self.crossover(population[i], population[i + 1])
                if len(t) > 2:
                    t = self.select_best(t, 2)
                new_pop.extend(t)

        # мутация
        for i in range(len(new_pop)):
            self.mutate(new_pop[i], self.mutation_p)

        # добавляем сохраненную элиту в популяцию
        for e in elite:
            if e not in new_pop:
                new_pop.append(e)

        # добавляем родителей для сохранения размера популяции
        lack = size - len(new_pop)
        if lack > 0:
            new_pop.extend(sample(population, lack))

        return new_pop

    def start(self, show_plot=True, print_rate=True, print_stats=True):
        """ Возвращает историю - последовательность популяций
        :rtype : list
        """
        print "=" * 40
        if self.extremum == 'max':
            et = u'максимума'
        else:
            et = u'минимума'
        print u'поиск ' + et + u" функции z = " + self.func_xy
        print u"\n{:=^30}".format(u' начальная популяция ')

        generation = 0
        population = Genetic.generate(self.size, search_field=self.search_field)
        best_before = best_after = self.rate_population(population, print_rate)
        self.same_best_counter = 0
        history = [population]

        while self.to_continue(generation, best_after, best_before):
            generation += 1
            print u"\n{:=^30}".format(u' поколение № ' + str(generation) + ' ')
            best_before = best_after
            size_before = len(population)

            try:
                population = self.evolve_step(population)
            except Exception as e:
                if e.message == 'convergence':
                    print u'вырождение популяции'
                break

            history.append(population)

            best_after = self.rate_population(population, print_rate)

            if print_stats:
                print u"{:-^30}\nразмер\t{}\n" \
                      u"кроссоверов {:.0f}% ({})\n" \
                      u"мутаций {:.0f}% ({})".format(u' статистика ',
                                                     len(population),
                                                     self.crossovers * 100.0 / (
                                                         size_before / 2),
                                                     self.crossovers,
                                                     self.mutations * 100.0 / size_before,
                                                     self.mutations)
            self.crossovers = self.mutations = 0

        if show_plot:
            self.show_plot(history)

        return history

    def elites(self, history, verbose=True):
        """ Возвращает лучших особей в истории
        history - история популяций
        """
        elites = []

        for population in history:
            elites.append(self.rate_population(population, verbose=False))

        if verbose:
            print '\n\n' + u"{:*^30}".format(u' лучшие особи ')
            Genetic.print_with_score(elites, self.fitness(elites))

        return elites

    def solution(self, elites):
        """ Возвращает результат - лучшую особь - в виде ((x,y),score)
        :param elites: лучшие особи в истории популяций
        """
        return elites[-1], self.fitness([elites[-1]])[0]

    # def draw_plot(self, history):
    #     fig = plt.figure()
    #     canvas = FigureCanvas(self, -1, self.figure)
    #
    #     ax = fig.add_subplot(111, projection='3d')
    #
    #     for population in history:
    #         xs = [point[0] for point in population]
    #         ys = [point[1] for point in population]
    #         zs = self.fitness(population)
    #         ax.scatter(xs, ys, zs, c=np.arange(len(population)))
    #
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('F(x, y)')
    #
    #     return canvas

    def show_plot(self, history, elites=[]):
        """Показывает популяции из истории.

        """
        colors = iter(cm.rainbow(np.linspace(0, 1, len(history))))

        for i, population in enumerate(history):
            xs = [point[0] for point in population]
            ys = [point[1] for point in population]
            zs = self.fitness(population)

            color = next(colors)
            self.ax.plot(xs, ys, zs, 'o', c=color, label=i, markersize=3, mew=0)

            # выделяем элиту в каждой популяции
            if len(history) == len(elites):
                self.ax.plot([elites[i][0]], [elites[i][1]], self.fitness([elites[i]]), 'x', c=color, markersize=20)

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=1, fontsize=8)
        plt.show()


class GeneticTester:
    def __init__(self, f_xy):
        self.function = f_xy

    @staticmethod
    def test_stohastic():
        f = 'x*x+y*y'
        g = Genetic(f, 'min', size=10)
        population = [(0.2, 0.5), (-0.2, 1), (0.2, 0.2), (-1, 0.3), (0.4, 0.8)]
        fit = g.fitness(population)
        print population
        Genetic.print_with_score(population, g.fitness(population))
        print 'avg fitness = ' + str(sum(fit) / float(len(fit)))
        print g.stochastic_sampling(population)

    def var_mp(self):
        elites = []
        scores = []
        for p in np.arange(0, 0.02, 0.003):
            g = Genetic(self.function, mp=p)
            best = g.start(show_plot=False)
            elites.append(best[0])
            scores.append(best[1])

        print '\n\n' + "{:$^30}".format(' elite ')
        Genetic.print_with_score(elites, scores)

    def many(self, n):
        elites = []
        scores = []
        g = Genetic(self.function)
        for i in range(n):
            best = g.start(show_plot=False)
            elites.append(best[0])
            scores.append(best[1])

        print '\n\n' + "{:$^30}".format(' elite ')
        Genetic.print_with_score(elites, scores)


if __name__ == '__main__':
    fxy = '(x*x + y*y)'
    # f = raw_input("Enter function of x and y: ")

    # tester = GeneticTester(f)
    # tester.test_stohastic()

    gen = Genetic(fxy, extremum='min')
    ghist = gen.start()
    gelite = gen.elites(ghist)
    gen.show_plot([gelite])
