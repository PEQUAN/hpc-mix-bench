# coding=utf-8

# This file is part of PROMISE.
#
# 	PROMISE is free software: you can redistribute it and/or modify it
# 	under the terms of the GNU Lesser General Public License as
# 	published by the Free Software Foundation, either version 3 of the
# 	License, or (at your option) any later version.
#
# 	PROMISE is distributed in the hope that it will be useful, but WITHOUT
# 	ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# 	or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General
# 	Public License for more details.
#
# 	You should have received a copy of the GNU Lesser General Public
# 	License along with PROMISE. If not, see
# 	<http://www.gnu.org/licenses/>.
#
# Promise v1 was written by Romain Picot
# Promise v2 has been written from v1 by Thibault Hilaire and Sara Hoseininasab
# Promise v3 has been written from v2 by Thibault Hilaire, Fabienne JÉZÉQUEL and Xinye Chen
#   Promise v3 enables version check, pip installing, arbitrary precision, etc., features.
# 	  Sorbonne Universitéx, LIP6 (Computing Science Laboratory), Paris, France. 
#     Contact: thibault.hilaire@lip6.fr, fabienne.jezequel@lip6.fr, xinyechenai@gmail.com
#
# 	contain the entry function, called to run Promise
#
# 	© Thibault Hilaire and Fabienne JÉZÉQUEL, April 2024



"""
\U0001f918 cadnaPromise \U0001f918

Usage:
	promise (-h | --help)
	promise (-v | --version)
	promise --precs=<strs> [options]

Options:
	-h --help                     Show this screen.
	--version                     Show version.
	--precs=<strs>                Set the precision following the built-in or cutomized precision letters [default: sd]
	--conf CONF_FILE              Get the configuration file [default: promise.yml]
	--fp FPT_FILE                 Get the file for floating point number format [default: fp.json]
	--output OUTPUT               Set the path of the output (where the result files are put)
	--verbosity VERBOSITY         Set the verbosity (betwen 0  and 4 for very low level debug) [default: 1]
	--log LOGFILE                 Set the log file (no log file if this is not defined)
	--verbosityLog VERBOSITY      Set the verbosity of the log file
	--debug                       Put intermediate files into `debug/` (and `compileErrors/` for compilation errrors) and display the execution trace when an error comes
	--run RUN                     File to be run
	--compile COMMAND             Command to compile the code
	--files FILES                 List of files to be examined by Promise (by default, all the .cc files)
	--nbDigits DIGITS             General required number of digits
	--path PATH                   Set the path of the project (by default, the current path)
	--pause                       Do pause between steps
	--noParsing                   Do not parse the C file (__PROMISE__ are replaced and that's all)
	--auto                        enable auto-instrumentation of source code
	--relError THRES              use criteria of precision relative error less than THRES instead of number of digits
	--noCadna                     will not use cadna, reference result computed in (non-stochastic) double precision
	--alias ALIAS                 Allow aliases (examples "g++=g++-14") [default:""]
	--CC        				  Set compiler for C program [default: g++]
	--CXX                         Set compiler for C++ program [default: g++]
"""


import os
from os.path import join
import sys
from collections import Counter
from docopt import docopt

from .utils import parseOptions, PromiseError, getYMLOptions, Timing, pause, commaAnd
from .utils import getFPM, sort_precs, update_types, get_version

from .logger import PrLogger


# types handle by cadnaPromise
_typeNames = {'b':'bfloat16', 
			  'h': 'Half', 
			  's': 'Single', 
			  'd': 'Double', 
			  'q': 'Quad', 
			  'o': 'Octuple'
			  }

_types = {'b': 'flx::floatx<8, 7>', 
		  'h': 'half_float::half', 
		  's': 'float', 
		  'd': 'double', 
		  'q': 'float128',
		  'o': 'flx::floatx<19, 236>',
		 }



def loadCADNA():
	if 'CADNA_PATH' in os.environ:
		print(os.environ['CADNA_PATH'])
	
	else:
		curr_loc = os.path.dirname(os.path.realpath(__file__))
		print(curr_loc+'/cadna')


def runPromise(argv=None):
	
	"""This function is registered (in setup.py) as an entry_point
	argv is used for the unit tests"""
	from .prfile import PrFile
	from .promise import Promise
	# reset the logger and get a new instance
	logger = PrLogger()
	
	# reset the handlers, in case of running runPromise several times (otherwise, the log files are still open)
	
	displayTrace = False
	EARLY_SROP = False
		
	try:
		if argv is None:
			if '--help' in sys.argv[1:] or len(sys.argv[1:]) == 0:
				print(__doc__)
				return
		else:
			if '--help' in argv or len(argv) == 0:
				print(__doc__)
				return

		if argv is not None:
			if '--version' in argv or '--v' in argv:
				EARLY_SROP = True

		else:
			if '--version' in sys.argv[1:] or '--v' in sys.argv[1:]:
				EARLY_SROP = True

		logger.reset()
		args = docopt(__doc__, argv=sys.argv[1:] if argv is None else argv)
		displayTrace = args['--debug']

		logger.configureLogger(args)   # configure the logger

		if EARLY_SROP:
			logger.message("cadnaPromise version " + get_version(
				os.path.dirname(os.path.realpath(__file__))+'/__init__.py') + ' (cadna version 3.1.12)')
	 
			logger.message("Copyright (c) 2024, GNU General Public License v3.0")
			logger.message(
				f"This work was supported by the France 2030 NumPEx Exa-MA (ANR-22-EXNU-0002) project managed by the French National Research Agency (ANR)."
				)
			
			return 
		

		options = getYMLOptions(args)                                           # get the options from the yml file

		method, path, files, run, nbDigits, _, compileLines, outputPath, typeCustom, alias = parseOptions(options)    # parse the options

		fpfmt = getFPM(args)
		method = sort_precs(method, fpfmt)

		types, typeNames = update_types(_types, _typeNames, fpfmt)
		compiler = 'g++'

		if isinstance(alias, dict):
			if alias == {}:
				curr_loc = os.path.dirname(os.path.realpath(__file__))

				cachePath = "/cache"
				if os.path.exists(curr_loc + cachePath):
					if os.path.isfile(curr_loc + cachePath + '/CXX.txt'):
						with open(curr_loc+cachePath+"/CXX.txt", "r") as file:
							compiler = file.read().replace('\n', '')
							print('check compilers:', compiler)
			
			elif alias.get('g++', False):
				compiler = alias['g++']

		else:
			compiler = alias
			alias = {}
			
		if compiler != 'g++' and compiler is not None:
			alias['g++'] = compiler
		

		logger.message("\U0001f918 cadnaPromise \U0001f918")
		logger.message("Using the compiler: {}".format(compiler))

		PrFile.setCustom(typeCustom)
		compileErrorPath = join(path, 'compileErrors') if args['--debug'] else None
		tempPath = join(path, 'debug') if args['--debug'] else None

		# run with timing
		with Timing() as timing:
			# create Promise object
			pr = Promise(path, files, run, nbDigits, compileLines, parsing=not args['--noParsing'], alias=alias)

			# display general infos
			logger.info("We are working with %d file%s and %d different types" %
			            (pr.nbFiles, ('' if pr.nbFiles < 2 else 's'), pr.nbVariables))
			logger.info(pr.expectation())

			# debug the files
			if args['--debug']:
				pr.exportParsedFiles(tempPath)

			# get the cadna reference
			highest = types['q'] if 'q' in method else types['d']
			# print("highest:", highest)
			pr.changeSameType(highest)
			logger.step("Get a reference result with cadna (%s)" % highest)
			pr.compileAndRun(tempPath, cadna=True)
			if args['--pause']:
				pause()

			# try with the highest precision
			logger.step("Check with highest format (%s)" % typeNames[method[-1]])
			pr.changeSameType(types[method[-1]])
			
			if not pr.compileAndRun(tempPath):
				pr.changeSameType(highest)
				raise PromiseError("You should lower your expectation, it doesn't work with " + typeNames[method[-1]])
			
			if args['--pause']:
				pause()
			

			# do the Delta-Debug passes ('s','d' and then 'h','s' when method is 'hsd' for example)
			for lo, hi in reversed(list(zip(method, method[1:]))):
				logger.step("Delta-Debug %s/%s" % (typeNames[lo], typeNames[hi]))
				res = pr.runDeltaDebug(types[lo], types[hi], tempPath, args['--pause'], compileErrorPath)
				# stop if the DeltaDebug is not successful
				if not res:
					break

		# export the output
		if argv is None:
			pr.exportFinalResult(outputPath)


	except PromiseError as e:
		logger.error(e, exc_info=displayTrace)

	else:
		if timing:
			# display the number of each type
			# count = Counter(pr.typesDict.values())  # count the nb of each type (result is a dictionary type:nb)
			# li = ["%dx %s" % (v, k) for k, v in count.items()]
			_output =  pr.setPerType()
			output = {types[i]:_output[types[i]] for i in fpfmt if types[i] in _output}
			output = dict(reversed(list(output.items())))
		
			li = ["%dx %s" % (len(output[i]), i) for i in output]

			logger.message("The final result contains %s.", commaAnd(li))
			logger.debug("Final types:\n" + pr.strResult())

			# display the stats
			logger.message("It tooks %.2fs", timing.timing)
			logger.message("\U0001F449 %d compilations (%d failed) for %.2fs", *pr.compilations)
			logger.message("\U0001F449 %d executions   (%d failed) for %.2fs", *pr.executions)

			logger.reset()

			return output
		

	logger.reset()






if __name__ == "__main__":
	runPromise()



