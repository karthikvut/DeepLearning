import numpy as np

a = np.random.randn(5)

print(a)

expa = np.exp(a)

print(expa)

answer = expa/expa.sum()

print(answer)

A = np.random.rand(100,5)

print(A)

expA = np.exp(A)

print(expA)

answerA = expA/expA.sum(axis=1,keepdims=True)

print(answerA)

print(expA.sum(axis=1,keepdims=True))