import numpy as numpy
import matplotlib.pyplot as pyplot


l = 1.7
m = 3.6

coef = numpy.array([[-l, m, 0, 0],
                   [l, -(m + l), 2 * m, 0],
                   [0, l, -(2 * m + l), 2 * m],
                   [0 + 1, 0 + 1, l + 1, -2 * m + 1]])
solutionArray = numpy.array([0, 0, 0, 1])
statProbArray = numpy.linalg.solve(coef, solutionArray)
print(statProbArray)

leftBorder = 0
rightBorder = 10
n = 1000
x = numpy.linspace(leftBorder, rightBorder, n)
h = (rightBorder - leftBorder) / n

S0 = []
S1 = []
S2 = []
S3 = []
workload = []
downtime = []

S0.append(1)
S1.append(0)
S2.append(0)
S3.append(0)
workload.append(0)
downtime.append(1)

for i in range(1, n):
    S0.append(S0[i - 1] + h * (-l * S0[i - 1] + m * S1[i - 1]))
    S1.append(S1[i - 1] + h * (l * S0[i - 1] - (l + m) * S1[i - 1] + 2 * m * S2[i - 1]))
    S2.append(S2[i - 1] + h * (l * S1[i - 1] - (2 * m + l) * S2[i - 1] + 2 * m * S3[i - 1]))
    S3.append(S3[i - 1] + h * (l * S2[i - 1] - 2 * m * S3[i - 1]))
    workload.append(S1[i] / 2 + S2[i] + S3[i])
    downtime.append(S0[i] + S1[i] / 2)

fig, axs = pyplot.subplots(2, 1, constrained_layout=True)
axs[0].plot(x, S0, '-r', x, S1, '-g',x, S2, '-b', x, S3, '-y')
axs[0].set_xlabel('t')
axs[0].set_ylabel('P')
axs[0].grid(True)
axs[1].plot(x, workload, '-r', x, downtime, '-g')
axs[1].set_xlabel('t')
axs[1].set_ylabel('coef')
axs[1].grid(True)

pyplot.show()
