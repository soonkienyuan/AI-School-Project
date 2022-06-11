import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow
import random as rd
import json

with open("cafe.json") as f:
	corspd= json.load(f)
patterns = [inst['patterns'] for  inst in corspd]
words = []
for pattern in patterns:
	wrds = nltk.word_tokenize(pattern)
	words.extend(wrds) # concatenate 2 list
words = list(set(words))
words = [w for w in words if w not in '?&,.']
words = sorted(words)


training = []
output = []
out_empty = [0 for _ in range(len(corspd))]

for i, pair in enumerate(corspd):
	response = pair['responses'][0]
	wrds = nltk.word_tokenize(response)
	bag = []
	wrds = [stemmer.stem(w) for w in wrds]
	for w in words:
		if w in wrds:
			bag.append(1)
		else:
			bag.append(0)
	output_row = out_empty[:]
	output_row[i] = 1

	training.append(bag)
	output.append(output_row)
training = np.array(training)
output = np.array(output)

# Buidling AI
tensorflow.compat.v1.reset_default_graph() # this is required to run afterwards
net = tflearn.input_data(shape=[None, len(training[0])])

net = tflearn.fully_connected(net, 8) # 8 neurons
net = tflearn.fully_connected(net, 8) # 8 neurons
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # output layers
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size = 8, show_metric = True)
model.save("model2.tflearn")