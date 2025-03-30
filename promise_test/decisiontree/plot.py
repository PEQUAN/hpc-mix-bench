import matplotlib.pyplot as plt

from cadnaPromise.run import runPromise
method = 'hsd'
digits = 12

testargs = ['--precs='+method, '--nbDigits=' + str(digits), '--conf=promise.yml' , '--fp=fp.json']
t = runPromise(testargs)