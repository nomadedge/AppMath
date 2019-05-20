import numpy as numpy
import matplotlib.pyplot as pyplot


class task2(object):
    def __init__(self, learning_rate = 0.1):
        self.weights_0_1 = numpy.random.normal(0.0, 1, (3, 3))
        self.weights_1_2 = numpy.random.normal(0.0, 1, (1, 3))
        self.sigmoid_mapper = numpy.vectorize(self.sigmoid)
        self.learning_rate = numpy.array([learning_rate])

    def func2(self, x, y, z):
        return((not(x or y)) == (not(z) or x))

    def sigmoid(self, x):
        return(1 / (1 + numpy.exp(-x)))

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
            weights_delta_layer_1).T  * self.learning_rate

def MSE(arr1, arr2):
    return(numpy.mean((arr1 - arr2) ** 2))

train = [
    ([0, 0, 0], 1),
    ([0, 0, 1], 0),
    ([0, 1, 0], 0),
    ([0, 1, 1], 1),
    ([1, 0, 0], 0),
    ([1, 0, 1], 0),
    ([1, 1, 0], 0),
    ([1, 1, 1], 0),
    ]
epochs = 5000
learning_rate = 0.1
loss = []

network = task2(learning_rate = learning_rate)

for e in range(epochs):
    inumpyuts_ = []
    correct_predictions = []
    for inumpyut_stat, correct_predict in train:
        network.train(numpy.array(inumpyut_stat), correct_predict)
        inumpyuts_.append(numpy.array(inumpyut_stat))
        correct_predictions.append(numpy.array(correct_predict))
    train_loss = MSE(network.predict(numpy.array(inumpyuts_).T), numpy.array(correct_predictions))
    loss.append(train_loss)

for inumpyut_stat, correct_predict in train:
    print("For inumpyut: {} the prediction is: {}, expected: {}".format(
        str(inumpyut_stat),
        str(network.predict(numpy.array(inumpyut_stat)) > .5),
        str(correct_predict == 1)))

print()

for inumpyut_stat, correct_predict in train:
    print("For inumpyut: {} the prediction is: {}, expected: {}".format(
        str(inumpyut_stat),
        str(network.predict(numpy.array(inumpyut_stat))),
        str(correct_predict == 1)))

pyplot.plot(loss)
pyplot.ylabel('MSE')
pyplot.xlabel('epoch')
pyplot.show()