import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import math

tf.logging.set_verbosity(tf.logging.INFO)


epoch = 1500
learningRate = 0.1
batch_size = 120

mnist_data = "/home/rohan/Downloads/MNIST_data"

trainForRandomSet = True


MNIST_DATASET = input_data.read_data_sets(mnist_data)

train_data = np.array(MNIST_DATASET.train.images, 'float32')
train_target = np.array(MNIST_DATASET.train.labels, 'int64')
print("training set consists of ", len(MNIST_DATASET.train.images), " instances")
test_data = np.array(MNIST_DATASET.test.images, 'float32')
test_target = np.array(MNIST_DATASET.test.labels, 'int64')
print("test set consists of ", len(MNIST_DATASET.test.images), " instances")


feature_columns = [tf.contrib.layers.real_valued_column("", dimension=len(MNIST_DATASET.train.images[1]))]
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, n_classes=10 , hidden_units=[128, 32]  , 
	optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=learningRate), activation_fn = tf.nn.relu, model_dir="model")


def generate_input_fn(data, label):	
		image_batch, label_batch = tf.train.shuffle_batch(
			[data, label] , batch_size=batch_size, capacity=8*batch_size, mindeque=4*batch_size, enqueue_many=True)
		return image_batch, label_batch
	
def input_fn_for_train():
	return generate_input_fn(train_data, train_target)
	
	#train on small random selected dataset
classifier.fit(input_fn=input_fn_for_train, steps=epoch)

print("\n---training is over...")

predictions = classifier.predict_classes(test_data)
index = 0
for i in predictions:
	if index < 10: #visualize first 10 items on test set
		print("actual: ", test_target[index], ", prediction: ", i)
		
		pred = MNIST_DATASET.test.images[index]
		pred = pred.reshape([28, 28]);
		plt.gray()
		plt.imshow(pred)
		plt.show()
		
	index  = index + 1


print("\n---evaluation...")
accuracy_score = classifier.evaluate(test_data, test_target, steps=epoch)['accuracy']
print("accuracy: ", 100*accuracy_score,"%")
