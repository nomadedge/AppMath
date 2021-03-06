import numpy as numpy
import matplotlib.pyplot as pyplot


def mark_modeling(trans, initialVector, accuracy, axs):
    newVector = initialVector
    oldVector = []
    standardDeviationArray = []
    stepArray = []

    def dot(oldVector, trans):
        newVector = numpy.dot(oldVector, trans)
        return newVector
    
    standardDeviationCurrent = accuracy
    stepCurrent = 1;

    while standardDeviationCurrent >= accuracy:
        oldVector = newVector
        newVector = dot(oldVector, trans)
        standardDeviationCurrent = numpy.std(newVector - oldVector)
        standardDeviationArray.append(standardDeviationCurrent)
        stepArray.append(stepCurrent)
        stepCurrent += 1

    print(newVector)
    axs.plot(stepArray, standardDeviationArray)
    axs.set_xlabel('Step')
    axs.set_ylabel('Standard deviation')


def mark_analysis(trans):
    trans = trans.transpose()
    trans -= numpy.identity(8);
    for i in range(8):
        trans[7][i] = 1
    solutionArray = numpy.array([0, 0, 0, 0, 0, 0, 0, 1])
    result = numpy.linalg.solve(trans, solutionArray)
    print(result)


accuracy = 1e-4
trans = numpy.array([[0.3, 0.2, 0, 0.1, 0, 0.4, 0, 0],
                     [0.1, 0.1, 0.5, 0.2, 0, 0, 0.1, 0],
                     [0.2, 0, 0.2, 0, 0.2, 0, 0, 0.4],
                     [0, 0.1, 0, 0.8, 0.1, 0, 0, 0],
                     [0, 0, 0.6, 0, 0.2, 0, 0.1, 0.1],
                     [0, 0, 0.3, 0.3, 0, 0.4, 0, 0],
                     [0, 0.2, 0.2, 0, 0, 0, 0.5, 0.1],
                     [0, 0.1, 0.1, 0, 0.1, 0, 0.1, 0.6]])
initialVector1 = numpy.array([1, 0, 0, 0, 0, 0, 0, 0])
initialVector2 = numpy.array([0, 1, 0, 0, 0, 0, 0, 0])

fig, axs = pyplot.subplots(2, 1, constrained_layout=True)
mark_modeling(trans, initialVector1, accuracy, axs[0])
mark_modeling(trans, initialVector2, accuracy, axs[1])
pyplot.show()

mark_analysis(trans)