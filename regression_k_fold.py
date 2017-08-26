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

	k_fold_mse = []
	test_mse = []
	x_axis =  range(1, 2500, 20)

	for a in range(1, 2500, 20):
		
		regr = linear_model.Ridge(alpha=a)
		regr.fit(train_features, train_target)

		test_predictions = regr.predict(test_features)
		test = mean_squared_error(test_target, test_predictions) / 100
		test_mse.append(test)

		k_fold = 1
		k_fold_acc = 1
		k = 15
		for x in range(1, k):
			k_fold_predictions, k_fold_target = predict_k_fold(data_set, a) 
			temp_fold = mean_squared_error(k_fold_target, k_fold_predictions) / 100
			if (fabs(test-temp_fold) < k_fold_acc):
				k_fold = temp_fold
				k_fold_acc = fabs(test-temp_fold)

		k_fold_mse.append(k_fold)

	plt.plot(x_axis, k_fold_mse, label='k_fold_data')
	plt.plot(x_axis, test_mse, label='test_data')

	plt.xlabel('alpha')
	plt.ylabel('Mean squared error')

	plt.legend()

	plt.show()

def predict_k_fold(data_set, _alpha):
	features, target = split_features_target(data_set)
	split = 0.15
	train_features, train_target, test_features, test_target = split_train_test(features, target, split)
	regr = linear_model.Ridge(alpha=_alpha)
	regr.fit(train_features, train_target)
	return regr.predict(test_features), test_target


def split_train_test(features, target, test_size):
	total_test_size = int(len(features) * test_size)
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