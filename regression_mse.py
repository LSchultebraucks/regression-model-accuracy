import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from math import fabs

def main():
	data_set = datasets.load_boston()

	features, target = split_features_target(data_set)

	split = 0.15
	train_features, train_target, test_features, test_target = split_train_test(features, target, split)

	train_mse = []
	test_mse = []
	x_axis =  range(1, 2500)

	min_train_mse, min_test_mse, min_difference, min_alpha = 1, 1, 1, 1

	for a in range(1, 2500):
		regr = linear_model.Ridge(alpha=a)
		regr.fit(train_features, train_target)

		train_predictions = regr.predict(train_features)
		test_predictions = regr.predict(test_features)

		train = mean_squared_error(train_target, train_predictions) / 100
		test = mean_squared_error(test_target, test_predictions) / 100

		train_mse.append(train)
		test_mse.append(test)

		difference = fabs(train-test)
		if (difference < min_difference):
			min_train_mse = train
			min_test_mse = test
			min_difference = difference
			min_alpha = a

	print('Minimal difference: ', min_difference)
	print('Alpha: ', min_alpha)
	print('MSE on training data', min_train_mse)
	print('MSE on test data', min_test_mse)

	plt.plot(x_axis, train_mse, label='train data')
	plt.plot(x_axis, test_mse, label='test_data')

	plt.xlabel('alpha')
	plt.ylabel('Mean squared error')

	plt.legend()

	plt.show()

def split_train_test(features, target, test_size):
	total_test_size = int(len(features) * test_size)
	np.random.seed(0)
	indices = np.random.permutation(len(features))
	train_features = features[indices[:-total_test_size]]
	train_target = target[indices[:-total_test_size]]
	test_features  = features[indices[-total_test_size:]]
	test_target  = target[indices[-total_test_size:]]
	return train_features, train_target, test_features, test_target

def split_features_target(data_set):
	features = data_set.data
	target = data_set.target
	return features, target

if __name__ == "__main__":
	main()