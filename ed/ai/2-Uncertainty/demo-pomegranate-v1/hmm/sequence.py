from pomegranate.markov_chain import *
import numpy
from model import model

# Observed data
observations = numpy.array([[[0],[0],[1],[0],[0],[0],[0],[1],[1]]]) # do I see an unbrella in a list of n individuals
print(observations.shape)

# Predict underlying states
predictions = model.predict(observations)
print(predictions)