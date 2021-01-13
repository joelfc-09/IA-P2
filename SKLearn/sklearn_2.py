import pandas as pd
from numpy import NaN

import Clustering.kmeans as km

# Preparar les dades

train_data = pd.read_csv("missing_data_train.csv")
test_data = pd.read_csv("missing_data_test.csv")

print(train_data.head())
print(test_data.head())

# Tractar les dades que falten

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer


def substituir_simple(data, strategy='constant', fill_value=0):
	# SimpleImputer
	imputer = SimpleImputer(missing_values=NaN, add_indicator=False, strategy=strategy, fill_value=fill_value)

	imputer.fit(data)

	return pd.DataFrame.from_records(imputer.transform(data), columns=data.columns)


def substituir_neighbours(data, num_neighbors=5, weights='uniform'):
	# kNNImputer
	imputer = KNNImputer(n_neighbors=num_neighbors, add_indicator=False, weights=weights)

	imputer.fit(data)

	return pd.DataFrame.from_records(imputer.transform(data), columns=data.columns)


# strategy constant: Canviar tots els missings per 1 sol valor
train_data_zeros = substituir_simple(train_data, strategy='constant', fill_value=0)
test_data_zeros = substituir_simple(test_data, strategy='constant', fill_value=0)

# strategy mean: Canviar tots els missings per la mitja de la columna
train_data_mean = substituir_simple(train_data, strategy='mean')
test_data_mean = substituir_simple(test_data, strategy='mean')

# strategy median: Canviar tots els missings per la mediana de la columna
train_data_median = substituir_simple(train_data, strategy='median')
test_data_median = substituir_simple(test_data, strategy='median')

# strategy most_frequent: Canviar tots els missings pel valor més repetit a la columna
train_data_most_frequent = substituir_simple(train_data, strategy='most_frequent')
test_data_most_frequent = substituir_simple(test_data, strategy='most_frequent')

# neighbours uniform: Canviar els missings per la mitjana dels veïns, amb pesos uniformes
train_data_uniform = substituir_neighbours(train_data)
test_data_uniform = substituir_neighbours(test_data)

# neighbours uniform: Canviar els missings per la mitjana dels veïns, amb pesos variables segons la distància
train_data_distance = substituir_neighbours(train_data, weights='distance')
test_data_distance = substituir_neighbours(test_data, weights='distance')

print(test_data.head())
print(test_data_zeros.head())
print(test_data_mean.head())
print(test_data_median.head())
print(test_data_most_frequent.head())
print(test_data_uniform.head())
print(test_data_distance.head())


# Entrenar una instància de Kmeans

def train_and_predict(kmeans, train_data, test_data):
	kmeans.fit(train_data.values.tolist())
	return kmeans.predict(test_data.values.tolist())


kmeans = km.Kmeans(5, distance=km.euclidean_squared, max_iters=5000, execution_times=100)

print("Zeros:\t\t" + str(train_and_predict(kmeans, train_data_zeros, test_data_zeros)))
print("Mean:\t\t" + str(train_and_predict(kmeans, train_data_mean, test_data_mean)))
print("Median:\t\t" + str(train_and_predict(kmeans, train_data_median, test_data_median)))
print("MostFreq:\t" + str(train_and_predict(kmeans, train_data_most_frequent, test_data_most_frequent)))
print("Uniform:\t" + str(train_and_predict(kmeans, train_data_uniform, test_data_uniform)))
print("Distance:\t" + str(train_and_predict(kmeans, train_data_distance, test_data_distance)))
