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

	holdout_mse = []
	test_mse = []
	x_axis =  range(1, 2500)

	for a in range(1, 2500):
		
		regr = linear_model.Ridge(alpha=a)
		regr.fit(train_features, train_target)

		holdout_predictions, holdout_target = predict_holdout(data_set, a)
		test_predictions = regr.predict(test_features)

		holdout = mean_squared_error(holdout_target, holdout_predictions) / 100
		test = mean_squared_error(test_target, test_predictions) / 100

		holdout_mse.append(holdout)
		test_mse.append(test)

	plt.plot(x_axis, holdout_mse, label='holdout_data')
	plt.plot(x_axis, test_mse, label='test_data')

	plt.xlabel('alpha')
	plt.ylabel('Mean squared error')

	plt.legend()

	plt.show()

def predict_holdout(data_set, _alpha):
	features, target = split_features_target(data_set)
	split = 0.15
	train_features, train_target, test_features, test_target = split_train_test(features, target, split)
	regr = linear_model.Ridge(alpha=_alpha)
	regr.fit(train_features, train_target)
	return regr.predict(test_features), test_target


def split_train_test(features, target, test_size, seed):
	total_test_size = int(len(features) * test_size)
	np.random.seed(seed)
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