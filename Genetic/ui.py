# -*- coding: utf-8 -*-

import wx
from genetic import Genetic
import sys
from random import random
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
import matplotlib.pyplot as plt

class PlotPanel(wx.Panel):
	def __init__(self,parent):
		wx.Panel.__init__(self, parent)
		self.figure = plt.figure()

		self.canvas = FigureCanvas(self,-1, self.figure)
		self.toolbar = NavigationToolbar(self.canvas)
		self.toolbar.Hide()

	def plot(self):
		data = [random() for i in range(25)]
		ax = self.figure.add_subplot(111, projection='3d')
		ax.hold(False)
		ax.plot(data, data, data, '*-')
		self.canvas.draw()

class RedirectText:
	def __init__(self,aWxTextCtrl):
		self.out = aWxTextCtrl

	def write(self,string):
		self.out.WriteText(string)

class GenUI(wx.Frame):
	PRECISION = 3
	row = 0

	def __init__(self, *args, **kwargs):
		super(GenUI, self).__init__(*args,  **kwargs)

		self.InitUI()
		self.SetDefaluts()
		self.Show(True)

	def InitUI(self):
		# функция
		self.row = 0
		gridFunction = wx.GridBagSizer(hgap=5, vgap=5)

		function_lbl = wx.StaticText(self, label=u"Функция от x и y:")
		self.function = wx.TextCtrl(self, size=(200,-1))

		self.extremum_rbox = wx.RadioBox(self, choices=['max', 'min'], label=u"Экстремум")

		self.AddRowToGridBag(gridFunction, function_lbl)
		self.AddRowToGridBag(gridFunction, self.function)
		self.AddRowToGridBag(gridFunction, self.extremum_rbox)

		self.extremum_rbox.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
		self.function.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

		# поле поиска
		gridSearchField = wx.GridBagSizer(hgap=5, vgap=5)
		boxSearchField = wx.StaticBoxSizer(wx.StaticBox(self, -1, u'Поле поиска'))
		boxSearchField.Add(gridSearchField)		
		self.AddRowToGridBag(gridFunction, boxSearchField)		

		x_lbl = wx.StaticText(self, label=u"x:")
		y_lbl = wx.StaticText(self, label=u"y:")
		from_lbl = wx.StaticText(self, label=u"от")
		to_lbl = wx.StaticText(self, label=u"до")

		self.minx = wx.TextCtrl(self, size=(50,-1))
		self.maxx = wx.TextCtrl(self, size=(50,-1))
		self.miny = wx.TextCtrl(self, size=(50,-1))
		self.maxy = wx.TextCtrl(self, size=(50,-1))

		self.row = 0
		self.AddRowToGridBag(gridSearchField, wx.StaticText(self), from_lbl, to_lbl)
		self.AddRowToGridBag(gridSearchField, x_lbl, self.minx, self.maxx)
		self.AddRowToGridBag(gridSearchField, y_lbl, self.miny, self.maxy)

		self.minx.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
		self.miny.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
		self.maxx.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
		self.maxy.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)


		# настройки
		self.row = 0
		gridGenetic = wx.GridBagSizer(hgap=5, vgap=5)

		# кроссовер
		cross_p_lbl = wx.StaticText(self, label=u"Вероятность кроссовера:")
		self.cross_p = wx.TextCtrl(self, size=(50,-1))
		self.cross_p_slider = wx.Slider(self, minValue=0, maxValue=10 ** GenUI.PRECISION, size=(100, -1), style=wx.SL_HORIZONTAL)

		self.AddRowToGridBag(gridGenetic, cross_p_lbl, self.cross_p, self.cross_p_slider)

		self.better_bind(wx.EVT_SCROLL, self.cross_p_slider, self.OnScroll, text_ctrl=self.cross_p, is_float=True)
		self.better_bind(wx.EVT_TEXT, self.cross_p, self.OnTextChanged, text_ctrl=self.cross_p, slider=self.cross_p_slider, is_float=True)

		self.cross_p.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

		# мутации
		mut_p_lbl = wx.StaticText(self, label=u"Вероятность мутации:")
		self.mut_p = wx.TextCtrl(self, size=(50,-1))
		self.mut_p_slider = wx.Slider(self, minValue=0, maxValue=10 ** GenUI.PRECISION / 10, size=(100, -1), style=wx.SL_HORIZONTAL)

		self.AddRowToGridBag(gridGenetic, mut_p_lbl, self.mut_p, self.mut_p_slider)

		self.better_bind(wx.EVT_SCROLL, self.mut_p_slider, self.OnScroll, text_ctrl=self.mut_p, is_float=True)
		self.better_bind(wx.EVT_TEXT, self.mut_p, self.OnTextChanged, text_ctrl=self.mut_p, slider=self.mut_p_slider, is_float=True)
		self.mut_p.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

		# размер начальный
		size_lbl = wx.StaticText(self, label=u"Начальный размер популяции:")
		self.size = wx.TextCtrl(self, size=(50,-1))
		self.size_slider = wx.Slider(self, minValue=10, maxValue=100, size=(100, -1), style=wx.SL_HORIZONTAL)

		self.AddRowToGridBag(gridGenetic, size_lbl, self.size, self.size_slider)

		self.better_bind(wx.EVT_SCROLL, self.size_slider, self.OnScroll, text_ctrl=self.size)
		self.better_bind(wx.EVT_TEXT, self.size, self.OnTextChanged, text_ctrl=self.size, slider=self.size_slider)
		self.size.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
		
		# элитизм
		elite_lbl = wx.StaticText(self, label=u"Оставлять лучших:")
		self.elite = wx.TextCtrl(self, size=(50,-1))		
		self.elite_slider = wx.Slider(self, minValue=0, maxValue=100, size=(100, -1), style=wx.SL_HORIZONTAL)
		
		self.AddRowToGridBag(gridGenetic, elite_lbl, self.elite, self.elite_slider)
		self.better_bind(wx.EVT_SCROLL, self.elite_slider, self.OnScroll, text_ctrl=self.elite)
		self.better_bind(wx.EVT_TEXT, self.elite, self.OnTextChanged, text_ctrl=self.elite, slider=self.elite_slider)
		self.elite.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

		# Blx
		blx_lbl = wx.StaticText(self, label=u"Blx alpha:")
		self.blx = wx.TextCtrl(self, size=(50,-1))		
		self.blx_slider = wx.Slider(self, minValue=0, maxValue=10 ** GenUI.PRECISION, size=(100, -1), style=wx.SL_HORIZONTAL)
		
		self.AddRowToGridBag(gridGenetic, blx_lbl, self.blx, self.blx_slider)
		self.better_bind(wx.EVT_SCROLL, self.blx_slider, self.OnScroll, text_ctrl=self.blx, is_float=True)
		self.better_bind(wx.EVT_TEXT, self.blx, self.OnTextChanged, text_ctrl=self.blx, slider=self.blx_slider, is_float=True)
		self.blx.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

		# отбор
		self.sampling_rbox = wx.RadioBox(self, choices=[str(x)[str(x).index('.') + 1:] for x in Genetic.SamplingType], style=wx.RA_VERTICAL, label=u"Отбор")
		self.crossover_rbox = wx.RadioBox(self, choices=[str(x)[str(x).index('.') + 1:] for x in Genetic.CrossoverType], style=wx.RA_VERTICAL, label=u"Кроссовер")	
	

		# число поколений
		maxgen_lbl = wx.StaticText(self, label=u"Максимум поколений:")
		self.maxgen = wx.TextCtrl(self, size=(50,-1))		
		self.maxgen_slider = wx.Slider(self, minValue=10, maxValue=1000, size=(100, -1), style=wx.SL_HORIZONTAL)
		
		self.AddRowToGridBag(gridGenetic, maxgen_lbl, self.maxgen, self.maxgen_slider)
		self.better_bind(wx.EVT_SCROLL, self.maxgen_slider, self.OnScroll, text_ctrl=self.maxgen)
		self.better_bind(wx.EVT_TEXT, self.maxgen, self.OnTextChanged, text_ctrl=self.maxgen, slider=self.maxgen_slider)
		self.maxgen.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

		# кнопки
		self.go_btn = wx.Button(self, label="Go")
		self.clear_btn = wx.Button(self, label=u"Очистить")		

		self.Bind(wx.EVT_BUTTON, self.OnGoClick, self.go_btn)
		self.Bind(wx.EVT_BUTTON, lambda x: self.log.Clear(), self.clear_btn)

		# флаги
		self.clear_chk = wx.CheckBox(self, label=u'Очищать при старте')
		self.rate_chk = wx.CheckBox(self, label=u'Особи в популяции')
		self.stats_chk = wx.CheckBox(self, label=u'Статистика')
		self.plot_chk = wx.CheckBox(self, label=u'График')
		
		# лог
		font = wx.Font(10, wx.MODERN, wx.NORMAL, wx.NORMAL, False, u'Consolas')
		self.log = wx.TextCtrl(self, size=(500,400), style = wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
		self.log.SetFont(font)
		redir = RedirectText(self.log)
		sys.stdout = redir

		# sizers
		sampling = wx.BoxSizer(wx.HORIZONTAL)
		sampling.Add(self.sampling_rbox)
		sampling.AddSpacer(10)
		sampling.Add(self.crossover_rbox)

		genetic = wx.BoxSizer(wx.VERTICAL)
		genetic.AddMany((gridGenetic, sampling))

		options = wx.FlexGridSizer(cols=2, hgap=10)
		options.AddMany((gridFunction, genetic))

		output_opt_box = wx.StaticBox(self, label=u'Вывод')
		output_opt = wx.StaticBoxSizer(output_opt_box, wx.VERTICAL)
		output_opt.AddMany((self.rate_chk, self.stats_chk, self.plot_chk, self.clear_chk))

		buttons = wx.BoxSizer(wx.HORIZONTAL)
		buttons.Add(self.go_btn, border=3)
		buttons.Add(output_opt)
		buttons.AddSpacer(15)
		buttons.Add(self.clear_btn, border=3)


		sizer = wx.BoxSizer(wx.VERTICAL)
		sizer.Add(options, 0, wx.EXPAND)
		sizer.Add(buttons, 0, wx.EXPAND)
		sizer.Add(self.log, 1, wx.EXPAND)

		#self.pp = PlotPanel(self)
		#sizer.Add(self.pp, 0, wx.EXPAND)


		# layout
		border = wx.BoxSizer()
		border.Add(sizer, 1, wx.ALL | wx.EXPAND, 5)
		self.SetSizerAndFit(border)  
		self.SetTitle(u'Поиск экстремума функции двух переменных')
		self.SetBackgroundColour((200,200,200))
		self.Centre()

	def SetDefaluts(self):
		self.mut_p.Value = '0.01'
		self.cross_p.Value = '0.9'
		self.size.Value = '30'
		self.function.Value = '100*(y-x^2)^2+(1-x)^2'# 'x*x+y*y'
		self.extremum_rbox.SetSelection(1)
		self.minx.Value = '0'
		self.miny.Value = '0'
		self.maxx.Value = '2'
		self.maxy.Value = '2'
		self.maxgen.Value = '50'
		self.elite.Value = '1'
		self.stats_chk.Value = True
		self.rate_chk.Value = True
		self.blx.Value = '0.5'


	def AddRowToGridBag(self, grid, *controls):
		for i, control in enumerate(controls):
			grid.Add(control, pos=(self.row, i))
		self.row += 1

	def better_bind(self, type, instance, handler, *args, **kwargs):
		self.Bind(type, lambda event: handler(event, *args, **kwargs), instance)

	def OnScroll(self, e, text_ctrl, is_float=False):
		if is_float:
			value = round(float(e.Position) / 10 ** GenUI.PRECISION, GenUI.PRECISION)
		else:
			value = e.Position
		text_ctrl.ChangeValue(str(value))
		if text_ctrl == self.size:
			self.elite_slider.SetMax(int(value))

	def OnTextChanged(self, e, text_ctrl, is_float=False, slider=None):
		if slider:
			try:
				value = float(text_ctrl.Value)
			except:
				value = 0
			if is_float:
				value = value * 10 ** GenUI.PRECISION
			slider.SetValue(int(value))
		if text_ctrl == self.size:
			self.elite_slider.SetMax(int(value))

	def OnKeyDown(self, e):
		key = e.GetKeyCode()
		if key == wx.WXK_RETURN:
			self.OnGoClick(None)
		e.Skip()

	def OnGoClick(self, e):
		if self.function.Value:
			if self.clear_chk.IsChecked():
				self.log.Clear()

			g = Genetic(functionXY = self.function.Value.encode('ascii','ignore'),
				extremum = self.extremum_rbox.GetStringSelection(),
				cp = float(self.cross_p.Value),
				mp = float(self.mut_p.Value),
				size = int(self.size.Value),				
				search_field = (float(self.minx.Value),
								float(self.miny.Value), 
								float(self.maxx.Value), 
								float(self.maxy.Value)),
				sampling = self.sampling_rbox.GetSelection(),
				crossover = self.crossover_rbox.GetSelection(),
				elite = int(self.elite.Value),
				max_generations = int(self.maxgen.Value),
				alpha = float(self.blx.Value))
			history = g.start(show_plot=False, printStats=self.stats_chk.IsChecked(), printRate=self.rate_chk.IsChecked())
			
			elites = g.elites(history)
			sol = g.solution(elites)

			print u'Найден экстремум\nF(x,y) = {}'.format(sol[1])
			print u'x = {}\ny = {}'.format(sol[0][0], sol[0][1])

			if self.plot_chk.IsChecked():
				g.showPlot(history, elites)

if __name__ == '__main__':
	ex = wx.App()
	GenUI(None)
	ex.MainLoop()
