import numpy as numpy
import matplotlib.pyplot as pyplot


class task1(object):
    def __init__(self, learning_rate=0.1):
        self.weights_0_1 = numpy.random.random_sample((100, 2)) * 2 - 1
        self.weights_1_2 = numpy.random.random_sample((1, 100)) * 2 - 1
        self.sigmoid_mapper = numpy.vectorize(self.sigmoid)
        self.lin_mapper = numpy.vectorize(self.lin)
        self.learning_rate = numpy.array([learning_rate])

    def func1(self, x):
        return (numpy.exp(numpy.arctan(x)))
        #return(numpy.log(1 + numpy.exp(x)))
        #return(numpy.exp(x)/numpy.cos(x))
        #return(numpy.log(1 + numpy.sin(x)))

    def sigmoid(self, x):
        return (1 / (1 + numpy.exp(-x)))

    def lin(self, x):
        return (x)

    def predict(self, inumpyuts):
        inumpyuts_1 = numpy.dot(self.weights_0_1, inumpyuts)
        outputs_1 = self.sigmoid_mapper(inumpyuts_1)
        inumpyuts_2 = numpy.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.lin_mapper(inumpyuts_2)
        return outputs_2

    def train(self, inumpyuts, expected_predict):
        inumpyuts_1 = numpy.dot(self.weights_0_1, inumpyuts)
        outputs_1 = self.sigmoid_mapper(inumpyuts_1)
        inumpyuts_2 = numpy.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.lin_mapper(inumpyuts_2)
        actual_predict = outputs_2

        error_layer_2 = numpy.array([actual_predict - expected_predict])
        gradient_layer_2 = numpy.ones(len(actual_predict))  # * (1 - actual_predict)
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2
        self.weights_1_2 -= (numpy.dot(weights_delta_layer_2,
                                       outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate

        error_layer_1 = weights_delta_layer_2 * self.weights_1_2
        gradient_layer_1 = outputs_1 * (1 - outputs_1)
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        self.weights_0_1 -= numpy.dot(inumpyuts.reshape(len(inumpyuts), 1),
                                      weights_delta_layer_1).T * self.learning_rate


class task2(object):
    def __init__(self, learning_rate=0.1):
        self.weights_0_1 = numpy.random.normal(0.0, 1, (3, 3))
        self.weights_1_2 = numpy.random.normal(0.0, 1, (1, 3))
        self.sigmoid_mapper = numpy.vectorize(self.sigmoid)
        self.learning_rate = numpy.array([learning_rate])

    def func2(self, x, y, z):
        return ((not (x or y)) == (not (z) or x))

    def sigmoid(self, x):
        return (1 / (1 + numpy.exp(-x)))

    def predict(self, inumpyuts):
        inumpyuts_1 = numpy.dot(self.weights_0_1, inumpyuts)
        outputs_1 = self.sigmoid_mapper(inumpyuts_1)
        inumpyuts_2 = numpy.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inumpyuts_2)
        return outputs_2

    def train(self, inumpyuts, expected_predict):
        inumpyuts_1 = numpy.dot(self.weights_0_1, inumpyuts)
        outputs_1 = self.sigmoid_mapper(inumpyuts_1)
        inumpyuts_2 = numpy.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inumpyuts_2)
        actual_predict = outputs_2[0]

        error_layer_2 = numpy.array([actual_predict - expected_predict])
        gradient_layer_2 = actual_predict * (1 - actual_predict)
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2
        self.weights_1_2 -= (numpy.dot(weights_delta_layer_2,
                                       outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate

        error_layer_1 = weights_delta_layer_2 * self.weights_1_2
        gradient_layer_1 = outputs_1 * (1 - outputs_1)
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        self.weights_0_1 -= numpy.dot(inumpyuts.reshape(len(inumpyuts), 1),
                                      weights_delta_layer_1).T * self.learning_rate


def MSE(arr1, arr2):
    return (numpy.mean((arr1 - arr2) ** 2))


train1 = [[]]
leftBorder = -10
rightBorder = 10
n = 1000
x = numpy.linspace(leftBorder, rightBorder, n)
epochs = 2000
learning_rate = 0.00045
loss = []
t1 = task1(learning_rate=learning_rate)
for val in x:
    f = t1.func1(val)
    buf = []
    buf.append(val)
    buf.append(f)
    train1.append(buf)

train1[0] = train1[1]

for e in range(epochs):
    print(e)
    inumpyuts_ = []
    correct_predictions = []
    for d in train1:
        inp = []
        inp.append(d[0])
        inp.append(1)
        t1.train(numpy.array(inp), d[1])
        inumpyuts_.append(numpy.array(inp))
        correct_predictions.append(numpy.array(d[1]))
    # a = numpy.random.random_sample() * 20 - 10
    # inp = []
    # inp.append(a)
    # inp.append(1)
    # b = t1.func1(a)
    # t1.train(numpy.array(inp), b)
    # inumpyuts_.append(numpy.array(inp))
    # correct_predictions.append(numpy.array(b))
    eps = 0.02
    n = 0
    ma = 0
    for d in range(0, 1000):
        h = numpy.random.random_sample() * 20 - 10
        inp = []
        inp.append(h)
        inp.append(1)
        di = numpy.abs(t1.predict(numpy.array(inp)) - t1.func1(h))
        if (di < eps):
            n += 1
        if (di > ma):
            ma = di
    print(n / 1000)
    print(ma)
    #train_loss = MSE(t1.predict(numpy.array(inumpyuts_).T), numpy.array(correct_predictions))
    loss.append(ma)

# eps = 0.02
# n = 0
# for d in range(0, 10000):
#     h = numpy.random.random_sample() * 20 - 10
#     inp = []
#     inp.append(h)
#     inp.append(1)
#     print(h, '\t', t1.predict(numpy.array(inp)), '\t', t1.func1(h))
#     if (numpy.abs(t1.predict(numpy.array(inp)) - t1.func1(h)) < eps):
#         n += 1
# print(n / 10000)
pyplot.plot(loss)
pyplot.ylabel('MSE')
pyplot.xlabel('epoch')
pyplot.show()

# train2 = [
#    ([0, 0, 0], 1),
#    ([0, 0, 1], 0),
#    ([0, 1, 0], 0),
#    ([0, 1, 1], 1),
#    ([1, 0, 0], 0),
#    ([1, 0, 1], 0),
#    ([1, 1, 0], 0),
#    ([1, 1, 1], 0),
#    ]
# epochs = 7000
# learning_rate = 0.12
# loss = []

# t2 = task2(learning_rate = learning_rate)

# for e in range(epochs):
#    inumpyuts_ = []
#    correct_predictions = []
#    for inumpyut_stat, correct_predict in train2:
#        t2.train(numpy.array(inumpyut_stat), correct_predict)
#        inumpyuts_.append(numpy.array(inumpyut_stat))
#        correct_predictions.append(numpy.array(correct_predict))
#    train_loss = MSE(t2.predict(numpy.array(inumpyuts_).T), numpy.array(correct_predictions))
#    loss.append(train_loss)

# for inumpyut_stat, correct_predict in train2:
#    print("For inumpyut: {} the prediction is: {}, expected: {}".format(
#        str(inumpyut_stat),
#        str(t2.predict(numpy.array(inumpyut_stat)) > .5),
#        str(correct_predict == 1)))

# print()

# for inumpyut_stat, correct_predict in train2:
#    print("For inumpyut: {} the prediction is: {}, expected: {}".format(
#        str(inumpyut_stat),
#        str(t2.predict(numpy.array(inumpyut_stat))),
#        str(correct_predict == 1)))

# pyplot.plot(loss)
# pyplot.ylabel('MSE')
# pyplot.xlabel('epoch')
# pyplot.show()