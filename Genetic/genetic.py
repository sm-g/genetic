# -*- coding: utf-8 -*-

from random import random, shuffle, sample, uniform
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from enum import IntEnum
from axiacore_parser import Parser

class Genetic():
	class SamplingType(IntEnum):
		stochastic = 0
		remainderStochastic = 1
		rank = 2
		tournament = 3

	class CrossoverType(IntEnum):
		simple = 0
		arithmetical = 1
		geometrical = 2
		BLXalpha = 3
		linear = 4

	parser = Parser()
	
	def __init__(self, functionXY, extremum='max', mp=0.02, cp=0.1, size=30, 
			  search_field=(-2,-2,2,2), sampling=SamplingType.stochastic, elite=0,
			  crossover=CrossoverType.simple, max_generations=1000, alpha=0.5):
		'''
		extremum - тип экстремума
		mp - вероятность мутации 
		cp - вероятность кроссовера 
		size - размер популяции
		search_field - поле поиска (minx, miny, maxx, maxy)
		sampling - тип отбора
		elite - кол-во лучших, переходящих в новое поколение
		crossover - тип размножения
		max_generations - максимальное количество поколений
		alpha - параметр для BLX кроссорвера
		'''
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
			Genetic.SamplingType.rank:					self.rankSampling,
			Genetic.SamplingType.tournament:			self.tournamentSampling,
			Genetic.SamplingType.stochastic:			self.stochasticSampling,
			Genetic.SamplingType.remainderStochastic:	self.remainderStochasticSampling}

		self.crossovers = 0
		self.mutations = 0
		self.functionXY = functionXY.lower()
		self.parsed_f = Genetic.parser.parse(self.functionXY)

		self.fig = plt.figure(0)
		self.ax = self.fig.add_axes([0,0,0.9,1], projection='3d')
		self.ax.set_xlabel('X')
		self.ax.set_ylabel('Y')
		self.ax.set_zlabel('F(x, y)')

	@staticmethod
	def generate(n, xmin=-0.5, ymin=-0.5, xsize=1, ysize=1, **kwargs):
		'''Генерирует случайную популяцию
		n - количество особей в популяции
		xmin, ymin- минимальные значения для x и y
		xsize, ysize - размеры поля, в котором генерируются координаты x и y
		kwargs
			Если передан search_field - поле поиска (minx, miny, maxx, maxy), 
			то используется для рассчета xmin, ymin, xsize, ysize 
		'''
		if kwargs['search_field']:
			xmin = kwargs['search_field'][0]
			ymin = kwargs['search_field'][1]
			xsize = kwargs['search_field'][2] - xmin
			ysize = kwargs['search_field'][3] - ymin
		points = [(random() * xsize + xmin, random() * ysize + ymin) for i in xrange(n)]
		return points

	def mutate(self, t, p, *args):
		'''Мутация особи. 
		t - точка
		p - вероятность мутации
		*args - dx и dy
		Меняет координаты точки на случайное число в диапазоне [-dc; dc]
		'''
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
	def BLX_gene(g1, g2, alpha):
		cmin = min(g1, g2)
		cmax = max(g1, g2)
		len = cmax - cmin
		return uniform(cmin - len * alpha, cmax + len * alpha)

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
	def BLX_crossover(t1, t2, alpha):
		ch1 = (Genetic.BLX_gene(t1[0], t2[0], alpha), Genetic.BLX_gene(t1[1], t2[1], alpha))
		return [ch1]
		
	def crossover(self, t1, t2):
		'''Cкрещивание двух особей с некоторой вероятностью'''

		if random() < self.crossover_p:
			self.crossovers += 1
			if self.crossover_type == Genetic.CrossoverType.simple:
				return Genetic.simple_crossover(t1,t2)
			elif self.crossover_type == Genetic.CrossoverType.arithmetical:				
				return Genetic.arithmetical_crossover(t1,t2,random())
			elif self.crossover_type == Genetic.CrossoverType.geometrical:				
				return Genetic.geometrical_crossover(t1,t2,random())
			elif self.crossover_type == Genetic.CrossoverType.linear:
				return Genetic.linear_crossover(t1,t2)
			elif self.crossover_type == Genetic.CrossoverType.BLXalpha:				
				return Genetic.BLX_crossover(t1,t2,self.alpha)
		return t1, t2
		
	def f(self, x, y):
		'''Заданная функция, для которой ищется экстремум'''
		try:
			return self.parsed_f.evaluate({'x': x, 'y': y})
		except Exception as e:
			print 'ошибка при вычислении функции: ' + str(e)

	def fitness(self, population):
		'''Приспособленность популяции.
		Возвращает массив значений целевой функции для каждой особи.
		'''
		return [self.f(ind[0], ind[1]) for ind in population]

	def fitness_normed(self, population):
		'''Нормированная приспособленность особей популяции'''
		fit = self.fitness(population)
		return Genetic.normalize_fitness(fit, self.extremum)
	
	def sorted_normed_fitness(self, population):
		'''Возвращает пары (номер особи в популяции, норма приспособленности), 
		отсортированные по норме.
		'''
		fit = self.fitness_normed(population)
		numbered_fit = [(i, x) for i, x in enumerate(fit)]
		numbered_fit.sort(key=lambda t: t[1])
		numbered_fit.reverse()
		return numbered_fit
		
	@staticmethod
	def normalize_fitness(fitness, extremum):
		'''Нормирует приспособленность от 0 до 1. Сумма приспособленностей особей популяции = 1.'''
		
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
	def printWithScore(population, score):
		print '\n'.join(['{}:\t{: .3f} {: .3f}\t{: .3f}'.format(i, t[0][0], t[0][1], t[1]) for i, t in enumerate(zip(population, score))])
		
	def rate_population(self, population, verbose=True):
		'''Оценка качества популяции. 
		Возвращает особь с максимальной приспособленностью.
		'''
		fit = self.fitness(population)
		score = Genetic.normalize_fitness(fit, self.extremum)
		maxScore = max(score)
		i = score.index(maxScore)
		if verbose:
			Genetic.printWithScore(population, score)		
			print u"{:-^30}\n{}:\t{: .3f} {: .3f}\t{:.3f}\nF(x,y) = {: .3f}".format(u' лучшая особь ', i, population[i][0], population[i][1], maxScore, fit[i])
		return (population[i][0], population[i][1])
		
	# отборы

	def remainderStochasticSampling(self, population):
		'''Пропорциональный отбор
		Для каждой особи вычисляется отношение ее приспособленности к средней 
		приспособленности популяции. Целая часть этого отношения указывает, 
		сколько раз нужно записать особь в промежуточную популяцию, 
		а дробная — это ее вероятность попасть туда еще раз. 
		'''
		score = self.fitness_normed(population)
		newpop = []
		for i, t in enumerate(population):
			r = score[i] * len(population)
			newpop.extend([t] * math.trunc(r))
			if random() < r - math.trunc(r):
				newpop.append(t)
		return newpop

	def stochasticSampling(self, population):
		'''Пропорциональный отбор 2
		Особи располагаются на колесе рулетки, так что размер сектора каждой особи 
		пропорционален ее приспособленности. Изначально промежуточная популяция пуста. 

		N раз запуская рулетку, выберем требуемое количество особей для записи 
		в промежуточную популяцию. Ни одна выбранная особь не удаляется с рулетки.
		'''
		score = self.fitness_normed(population)
		
		rights = [] # правые границы отрезков, заполняющих единичный отрезок рулетки
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

	def rankSampling(self, population):
		'''Ранковый отбор
		Для каждой особи вероятность попасть в промежуточную популяцию 
		пропорциональна ее порядковому номеру в отсортированном списке
		приспособленности популяции.
		'''
		
		rights = [] # правые границы отрезков, заполняющих единичный отрезок рулетки
		# для популяции из 3 особей длины отрезков: 3/6, 2/6, 1/6, границы: 1/6, 1/2,
					# 1
		ss = 0
		size = len(population) * (len(population) + 1) / 2.0
		for i in range(len(population)):
			ss +=(i + 1) / size
			rights.append(ss)

		newpop = []
		for t in self.sorted_normed_fitness(population):
			r = random()
			for i in range(len(population)):
				if r < rights[i]:
					newpop.append(population[t[0]])
					break
		return newpop

	def tournamentSampling(self, population):
		'''Турнирный отбор
		Из популяции случайным образом выбирается t особей, и лучшая из них 
		помещается в промежуточную популяцию. Этот процесс повторяется N раз, 
		пока промежуточная популяция не будет заполнена. 
		'''
		newpop = []
		for i in range(len(population)):
			while True:
				r = sample(population, 2) # выбираем 2 разных особи
				if r[0] != r[1]:
					break

			newpop.append(self.rate_population(r, verbose=False))

		return newpop

	def toContinue(self, generation, best, bestPrev):
		'''Условие продолжения поиска решения.
		generation - номер поколения
		best - лучшая особь в текущем поколении
		bestPrev - лучшая особь в предыдущем поколении
		'''		
		if generation == self.max_generations:
			print u"достигнут предел количества поколений"
			return False
		
		if best == bestPrev:
			self.same_best_counter += 1
		else:
			self.same_best_counter = 0

		if self.same_best_counter == 10:
			print u"приспособленность лучшей особи не меняется"
			return False

		return True
		
	def selectBest(self, population, n):
		'''Возвращает n лучших особей популяции
		'''
		if n <= 0:
			return []
		return [population[x[0]] for x in self.sorted_normed_fitness(population)][:n]

	@staticmethod
	def convergence(population):
		'''Условие вырождения популяции - все особи одинаковые'''
		return population.count(population[0]) == len(population)

	def evolve_step(self, population):
		'''Шаг эволюции
		Отбор (селекция). Скрещивание (размножение). Мутация.
		'''
		# элита
		elite = self.selectBest(population, self.elite)
		
		# промежуточная популяция после отбора - те, кто будет размножаться
		population = self.samplingDict[self.sampling](population)

		if Genetic.convergence(population):
			raise Exception('convergence')
		
		# перемешиваем для случайного разбиения на пары для размножения
		shuffle(population)

		new_pop = []
		size = len(population)
		
		# кроссовер
		for i in range(0,size,2):
			if i + 1 != size:
				t = self.crossover(population[i], population[i + 1])
				if len(t) > 2:
					t = self.selectBest(t, 2)
				new_pop.extend(t)

		# мутация
		for i in range(len(new_pop)):
			self.mutate(new_pop[i], self.mutation_p)	

		# добавляем сохраненную элиту в популяцию
		for e in elite:
			if not e in new_pop:
				new_pop.append(e)

		# добавляем родителей для сохранения размера популяции
		lack = size - len(new_pop)
		if lack > 0:
			new_pop.extend(sample(population, lack))

		return new_pop

	def start(self, show_plot=True, printRate=True, printStats=True):
		''' Возвращает историю - последовательность популяций
		'''
		print "=" * 40
		if self.extremum == 'max':
			et = u'максимума'
		else:
			et = u'минимума'
		print u'поиск ' + et + u" функции z = " + self.functionXY
		print u"\n{:=^30}".format(u' начальная популяция ')
		
		generation = 0
		population = Genetic.generate(self.size, search_field=self.search_field)
		bestBefore = bestAfter = self.rate_population(population, printRate)
		self.same_best_counter = 0		
		history = [population]

		while(self.toContinue(generation, bestAfter, bestBefore)):
			generation += 1
			print u"\n{:=^30}".format(u' поколение № ' + str(generation) + ' ')
			bestBefore = bestAfter
			sizeBefore = len(population)

			try:
				population = self.evolve_step(population)
			except Exception as e:
				if e.message == 'convergence':
					print u'вырождение популяции'
				break

			history.append(population)
			
			bestAfter = self.rate_population(population, printRate)
			
			if printStats:				
				print u"{:-^30}\nразмер\t{}\nкроссоверов {:.0f}% ({})\nмутаций {:.0f}% ({})".format(u' статистика ', len(population), self.crossovers * 100.0 / (sizeBefore / 2), self.crossovers, self.mutations * 100.0 / sizeBefore, self.mutations)
			self.crossovers = self.mutations = 0

		if show_plot:
			self.showPlot(history)
		
		return history

	def elites(self, history, verbose=True):
		''' Возвращает лучших особей в истории
		history - история популяций
		'''
		elites = []

		for population in history:
			elites.append(self.rate_population(population, verbose=False))

		if verbose:
			print '\n\n' + u"{:*^30}".format(u' лучшие особи ')
			Genetic.printWithScore(elites, self.fitness(elites))

		return elites

	def solution(self, elites):
		''' Возвращает результат - лучшую особь - в виде ((x,y),score)
		elites - лучшие особи в истории популяций
		'''
		return elites[-1], self.fitness([elites[-1]])[0]

	def drawPlot(self, history):
		fig = plt.figure()
		canvas = FigureCanvas(self,-1, self.figure)
		
		ax = fig.add_subplot(111, projection='3d')
		
		for population in history:
			xs = [point[0] for point in population]
			ys = [point[1] for point in population]
			zs = self.fitness(population)
			ax.scatter(xs, ys, zs, c=np.arange(len(population)))

		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('F(x, y)')

		return canvas
		
	def showPlot(self, history, *args):
		'''Показывает популяции из истории. 
		*args - список лучших в популяциях для выделения
		'''
		colors = iter(cm.rainbow(np.linspace(0, 1, len(history))))
		with_elites = len(args) > 0 and len(args[0]) == len(history)
		if with_elites:
			elites = args[0]

		for i, population in enumerate(history):
			xs = [point[0] for point in population]
			ys = [point[1] for point in population]
			zs = self.fitness(population)

			color = next(colors)
			self.ax.plot(xs, ys, zs, 'o', c=color, label=i, markersize=3, mew=0)

			if with_elites:
				self.ax.plot([elites[i][0]], [elites[i][1]], self.fitness([elites[i]]), 'x', c=color, markersize=20)		

		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), numpoints=1, fontsize=8)		
		plt.show()

class GeneticTester():
	def __init__(self, functionXY):
		self.function = functionXY

	def test_stohastic(self):
		f = 'x*x+y*y'
		g = Genetic(f, 'min', size=10)
		population = [(0.2, 0.5), (-0.2, 1), (0.2, 0.2), (-1, 0.3), (0.4, 0.8)]
		fit = g.fitness(population)
		print population
		Genetic.printWithScore(population, g.fitness(population))
		print 'avg fitness = ' + str(sum(fit) / float(len(fit)))
		print g.stochasticSampling(population)
	
	def var_mp(self):
		elites = []
		scores = []
		for p in np.arange(0,0.02,0.003):
			g = Genetic(self.function, mp=p)
			best = g.start(show_plot=False)
			elites.append(best[0])
			scores.append(best[1])

		print '\n\n' + "{:$^30}".format(' elite ')
		Genetic.printWithScore(elites, scores)
		
	def many(self, n):
		elites = []
		scores = []
		g = Genetic(self.function)
		for i in range(n):			
			best = g.start(show_plot=False)
			elites.append(best[0])
			scores.append(best[1])

		print '\n\n' + "{:$^30}".format(' elite ')
		Genetic.printWithScore(elites, scores)

if __name__ == '__main__':
	f = '(x*x + y*y)'
	#f = raw_input("Enter function of x and y: ")

	tester = GeneticTester(f)
	tester.test_stohastic()

	#gen = Genetic(f, extremum='min')
	#history = gen.start()
	#elites = gen.elites(history)
	#gen.showPlot([elites])
