# coding=utf-8

"""This file is part of PROMISE.

	PROMISE is free software: you can redistribute it and/or modify it
	under the terms of the GNU Lesser General Public License as
	published by the Free Software Foundation, either version 3 of the
	License, or (at your option) any later version.

	PROMISE is distributed in the hope that it will be useful, but WITHOUT
	ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
	or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General
	Public License for more details.

	You should have received a copy of the GNU Lesser General Public
	License along with PROMISE. If not, see
	<http://www.gnu.org/licenses/>.

Promise v1 was written by Romain Picot
Promise v2 has been written from v1 by Thibault Hilaire and Sara Hoseininasab
Promise v3 has been written from v2 by Thibault Hilaire and Fabienne JÉZÉQUEL and Xinye Chen
	Promise v3 enables version check, pip installing, arbitrary precision, etc., features.
	Sorbonne Université
	LIP6 (Computing Science Laboratory)
	Paris, France. Contact: thibault.hilaire@lip6.fr, fabienne.jezequel@lip6.fr, xinyechenai@gmail.com

Some useful functions to parse the C/C++ files and replace the __PROMISE__ and __PR_xxx__ by some types.
To do that, we have to parse the file, in order to know if the __PROMISE__ are in a comment, string, etc. or
if it is used for a variable declaration, an array, a pointer, a function returns, a function argument, etc.
Depending on the case, the code may be a little bit changed
(`__PROMISE__ x,y=5;` is changed in `__PROMISE__ x; __PROMISE__ y; y=5;` for example)


© Thibault Hilaire and Fabienne JÉZÉQUEL, April 2024
"""


from .dd import DD
from ..errors import PromiseCompilationError
from ..utils import pause
from ..logger import PrLogger

logger = PrLogger()


class PromiseDD(DD):
	"""Promise Delta Debug"""

	def __init__(self, pr, type1, type2, status, bar, path='', doPause=False, compileErrorPath=None):
		"""Create a Delta Debug object, with promise-dedicated test method
		Parameter:
		- pr: the Promise object of the project
		- type1, type2: (string) the two types uses for the delta-debug method
		(all the types initialy equal to types2 can be changed to type1)
		- status, bar: two tqdm objects (status and progress bar)
		- path: (string) path where put the generated files (compiled, run and tested)
			if empty, a temporary folder is used
		- doPause: (boolean) True if we pause after each dd iteratio"""
		self._status = status
		self._bar = bar
		self._type1 = type1
		self._type2 = type2
		self._pr = pr
		self._path = path
		self._pause = doPause
		self._initialTypes = dict(pr.typesDict)
		self._compileErrorPath = compileErrorPath
		super().__init__()

	def test(self, C):
		"""Test if the configuration
		Returns PASS, FAIL, or UNRESOLVED."""
		logger.debug('We test C=' + str(C))
		# change the types according to C
		self._pr.changeTypes(self._initialTypes)
		self._pr.changeSomeTypes(self._type1, C)
		# compile, run and get the result
		try:
			check = self._pr.compileAndRun(self._path, compilationErrorAsDebug=True, compileErrorPath=self._compileErrorPath)
			if self._pause:
				pause(self._status)
		except PromiseCompilationError:
			logger.debug('Compilation failed \U0001F44E')
			self._status.set_description_str('Compilation failed \U0001F44E')
			self._status.refresh()

			return DD.FAIL
		# update the status
		self._status.set_description_str('We test C=' + str(C) + (" \U0001F44D" if check else " \U0001F44E"))
		self._bar.update()
		return DD.PASS if check else DD.FAIL


	def run(self):
		"""run the Delta Debug algorithm
		Retuns what ddmax returns"""
		# run the dd
		deltas = self._pr.getTypesEqualTo(self._type2)
		t = self.ddmax(deltas)
		# set progress bar to 100%
		self._bar.total = self._bar.n
		self._bar.refresh()
		# return to the types given by dd algorithm
		self._pr.changeTypes(self._initialTypes)
		self._pr.changeSomeTypes(self._type1, list(set(deltas)-set(t)))
		logger.debug("DD algorithms returns : " + str(self._pr.typesDict))




class PromisePicireDD():
	"""PROMISE Delta-Debug using Picire's Delta-Debug in order to use parallelization (Hodovan & Kiss - https://github.com/renatahodovan/picire)"""

	def __init__(self, pr, type1, type2, status, bar, path='', doPause=False, compileErrorPath=None):
		"""Create a Delta Debug object, with promise-parallel-dedicated test method
		Parameter:
		- pr: the Promise object of the project
		- type1, type2: (string) the two types uses for the delta-debug method
		(all the types initially equal to types2 can be changed to type1)
		- status, bar: two tqdm objects (status and progress bar)
		- path: (string) path where put the generated files (compiled, run and tested)
			if empty, a temporary folder is used
		- doPause: (boolean) Ignored in this version of Delta-Debug"""
		import picire
		
		self._status = status
		self._bar = bar
		self._type1 = type1
		self._type2 = type2
		self._pr = pr
		self._path = path
		self._pause = doPause
		self._initialTypes = dict(pr.typesDict)
		self._compileErrorPath = compileErrorPath

		



	def run(self, config, proc_n):
		"""run Delta Debug parallel taken from Picire (Hodovan & Kiss - https://github.com/renatahodovan/picire)"""
		if proc_n == 1:
			dd_obj = picire.DD(self.test,  subset_iterator=picire.config_iterators.skip)
		else:
			dd_obj = picire.ParallelDD(self.test, proc_num=proc_n, subset_iterator=picire.config_iterators.skip)
		t = [config[x] for x in dd_obj(list(range(len(config))))]

		#out_bar = [x for x in deltas if x not in t]

		# return to the types given by dd algorithm
		self._pr.changeTypes(self._initialTypes)
		deltas = self._pr.getTypesEqualTo(self._type2)
		self._pr.changeSomeTypes(self._type1, list(set(deltas) - set(t)))
		return t

	def test(self, C, C_id):
		"""Test if the configuration
            Returns PASS, FAIL (or UNRESOLVED)."""
		# logger.debug('We test C=' + str(C))
		# change the types according to C
		self._pr.changeTypes(self._initialTypes)
		CC = self._pr.getTypesEqualTo(self._type2)
		C_var = [CC[i] for i in C]

		Cbar = [x for x in CC if x not in C_var]
		self._pr.changeSomeTypes(self._type1, Cbar)
		# compile, run and get the result

		check = self._pr.compileAndRun(self._path, compilationErrorAsDebug=True,
									 compileErrorPath=self._compileErrorPath)
		# update the status
		self._status.set_description_str('We test C=' + str(Cbar) + (" \U0001F44D" if check else " \U0001F44E"))
		self._bar.update()

		return picire.Outcome.FAIL if check else picire.Outcome.PASS
