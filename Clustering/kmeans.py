import random
import matplotlib.pyplot as plt
from functools import reduce
from tqdm import tqdm
import matplotlib.colors as plc
from time import sleep

filename = "seeds.csv"


def euclidean_squared(p1, p2):
	return sum(
		(val1 - val2) ** 2
		for val1, val2 in zip(p1, p2)
	)


def read_csv(csv_filename):
	rows = []
	try:
		with open(csv_filename, "r") as f:
			for line in [ln.rstrip('\n') for ln in f.readlines()]:
				rows.append([float(x) for x in line.split(",")])
	except FileNotFoundError:
		return []
	return rows


def create_chart(ks, data):
	plt.rcdefaults()

	num_ks = [i for i in range(1, ks)]
	bar_ch = plt.bar(num_ks, data)

	# Intra cluster variation (WSS):
	plt.title("Intra cluster variation sum by cluster number")

	plt.show()


class Kmeans:
	def __init__(self, k, distance, max_iters, use_range=True, execution_times=5):
		self.k = k  # Número de clusters
		self.distance = distance  # Funció de distància que farem servir
		self.use_range = use_range
		self.max_iters = max_iters
		self.inertia_ = 0
		self.centroids = []
		self.execution_times = execution_times
		self.best_matches = None

	# Crear un centroide random
	def _get_range_random_value(self, points, feature_idx):
		feat_values = [point[feature_idx] for point in points]  # Valors de la feature
		feat_max = max(feat_values)
		feat_min = min(feat_values)
		return random.random() * (feat_max - feat_min) + feat_min

	# Crear centroides random
	def _create_random_centroids(self, rows):
		n_feats = len(rows[0])
		self.centroids = []
		for cluster_idx in range(self.k):  # Per el número de clusters que volem
			point = [0.0] * n_feats
			for feature_idx in range(n_feats):  # Crear un centroid passant per tots els features
				point[feature_idx] = self._get_range_random_value(rows, feature_idx)
			self.centroids.append(point)

	def _create_points_centroids(self, points):
		raise NotImplementedError

	# Row es una llista que representa un punt
	# Busquem els centroide més proper al punt i retorna el id
	def _find_closest_centroid(self, row):
		min_dist = 2 ** 64
		closest_centroid_idx = None

		for centroid_idx, centroid in enumerate(self.centroids):  # self.centroids es una llista amb els centroides
			dist = self.distance(row, centroid)  # Distància del punt al centroide segons la funció que hem passat
			if dist < min_dist:
				closest_centroid_idx = centroid_idx
				min_dist = dist

		return closest_centroid_idx

	# Retornar el punt mitjana de points_in_cl, que es els punts que hi ha a un cluster
	def _average_points(self, points_in_cl):
		avgs = []
		for i in range(len(points_in_cl[0])):
			avgs.append(sum([p[i] for p in points_in_cl]) / float(len(points)))
		return avgs

	# matches es una llista on per cada cluster hi ha els ids dels punts que li toquen
	# rows es la llista amb tots els punts
	# Actualitzar els centroides dels clusters
	def _update_centroids(self, matches, rows):
		for cluster_idx in range(len(matches)):  # Per cada cluster
			points_2 = [rows[i] for i in matches[cluster_idx]]  # Agafo els punts enlloc dels ids
			if not points_2:  # Si no hi ha punts no hi ha centre
				continue
			avrg = self._average_points(points_2)
			self.centroids[cluster_idx] = avrg  # El nou centroide del cluster es la mitjana de tots els punts

	def fit(self, rows):

		total_bestmatches = None
		best_inertia = 0

		for _ in range(self.execution_times):

			if self.use_range:
				self._create_random_centroids(rows)  # Crear centroides random el primer cop
			else:
				self._create_points_centroids(rows)

			lastmatches = None  # Assignacions anteriors

			bestmatches = None

			for iteration in range(self.max_iters):  # Limit d'iteracions per evitar bucle infinit
				self.inertia_ = 0
				bestmatches = [[] for _ in range(self.k)]  # Matches buits al principi

				for row_idx, row in enumerate(rows):  # Per tots els punts
					centroid_idx = self._find_closest_centroid(row)  # Trobem el centroide mes proper i l'afegim
					# print(len(bestmatches), self.k, len(self.centroids))
					bestmatches[centroid_idx].append(row_idx)
					self.inertia_ += euclidean_squared(row, self.centroids[centroid_idx])

				# Si anteriors i actuals son iguals hem acabat
				if bestmatches == lastmatches:
					break

				lastmatches = bestmatches

			if total_bestmatches is None or self.inertia_ < best_inertia:
				total_bestmatches = bestmatches
				best_inertia = self.inertia_

		self.inertia_ = best_inertia
		# Actualitzar els centroides dels clusters segons les noves assignacions
		self._update_centroids(total_bestmatches, rows)
		self.best_matches = total_bestmatches
		return total_bestmatches

	# Passat una llista de punts retornar de quin cluster serien
	def predict(self, rows):
		predictions = list(map(self._find_closest_centroid, rows))
		return predictions

	def _a(self, rows, row_idx, cluster_idx):
		if len(self.best_matches[cluster_idx]) == 1:
			return 0
		idxs_of_cluster = [euclidean_squared(rows[row_idx], rows[x]) for x in self.best_matches[cluster_idx] if
		                   x != row_idx]
		sum_of_same_cluster_dists = sum(idxs_of_cluster)
		return (sum_of_same_cluster_dists) / (len(self.best_matches[cluster_idx]) - 1)

	def _b(self, rows, row_idx, cluster_idx):
		dists = []
		for clust_idx in range(len(self.best_matches)):
			if clust_idx == cluster_idx:
				dists.append(9999)
			elif len(self.best_matches[clust_idx]) == 0:
				dists.append(9999)
			else:
				distance_to_clust_sum = sum(
					[euclidean_squared(rows[row_idx], rows[x]) for x in self.best_matches[clust_idx]])
				avg_dist = distance_to_clust_sum / len(self.best_matches[clust_idx])
				dists.append(avg_dist)
		return reduce((lambda x, y: min(x, y)), dists, 9999)

	def silhouette(self, rows, row_idx):
		cluster_idx = [i for i in range(len(self.best_matches)) if row_idx in self.best_matches[i]][0]
		a = self._a(rows, row_idx, cluster_idx)
		b = self._b(rows, row_idx, cluster_idx)
		if max(a, b) == 0:
			return 0
		return (b - a) / max(a, b)

	def _intra_cluster_variation(self, rows, cluster_idx):
		total = 0
		for elem_idx, elem_row_idx in enumerate(self.best_matches[cluster_idx]):
			for elem2_row_idx in self.best_matches[cluster_idx][elem_idx:]:
				total += euclidean_squared(rows[elem_row_idx], rows[elem2_row_idx])
		return total

	def intra_cluster_variation(self, rows):
		return sum([self._intra_cluster_variation(rows, i) for i in range(len(self.best_matches))])

	def elbow_method(k, wss):
		# rows: els punts
		# wss: intra-cluster variation de tots els clusters (suma de distancies entre els punts del cluster, sumats tots els clusters)
		# Mètode: generar un gràfic amb els wss de totes les ks, el millor número de k és on "s'aplana" la curva, on baixe mes
		create_chart(k, wss)


def create_silhouette_chart(ks, silhouettes, chart=True):
	accepted_colors = ['darkorange', 'forestgreen', 'cornflowerblue', 'red', 'black', 'mediumorchid', 'darkolivegreen',
	                   'chocolate', 'crimson', 'deepskyblue', 'indigo', 'dimgray', 'lime', 'mediumspringgreen', 'gold']

	accepted_colors = ['black'] * 15

	accepted_colors_rgba = [plc.to_rgb(c) for c in accepted_colors]

	plt.rcdefaults()

	avg_sil = []

	for k_total in range(1, ks):
		dades = silhouettes[k_total]
		y_dades_total = []
		colors = []
		for k_actual in range(max(dades.keys()) + 1):
			try:
				y_dades = list(dades[k_actual])
			except KeyError:
				continue
			y_dades.sort(reverse=True)
			y_dades_total.extend(y_dades)
			colors.extend(accepted_colors_rgba[k_actual] * len(y_dades))
		# print(x_pos)
		# print(y_dades)
		x_pos_total = [x for x in range(len(y_dades_total))]
		if chart:
			plt.bar(x_pos_total, y_dades_total)
			plt.ylabel("Silhouette dels punts")
			titol = "Total de clusters: " + str(k_total) + " Average silhouette: " + str(
			sum(y_dades_total) / len(y_dades_total))
		avg_sil.append(sum(y_dades_total) / len(y_dades_total))
		if chart:
			plt.title(titol)
			plt.show()
	return avg_sil

# plt.show()

points = [
	[1, 1],
	[2, 1],
	[4, 3],
	[5, 4]
]

points2 = [
	[2, 4],
	[3, 5],
	[3, 2],
	[5, 2],
	[5, 4],
	[7, 3],
	[7, 8],
	[8, 4],
]

centroids = [
	[1, 1],
	[2, 1]
]

def apartat_clustering(filename="seeds.csv", csv_data_2=None):
	inertias = []
	silhouettes = {}
	wss = []
	k = 11
	if csv_data_2 is None:
		csv_data = read_csv(filename)
	else:
		csv_data = csv_data_2
	for valor_k in tqdm(range(1, k)):
		kmeans = Kmeans(k=valor_k, distance=euclidean_squared, max_iters=1000, execution_times=150)
		bestmatches2 = kmeans.fit(csv_data)

		inertias.append(kmeans.inertia_)

		# Cada cluster: Silhouette de cada punt agrupada per clusters, per fer el gràfic
		silhouettes_of_all_points = [kmeans.silhouette(csv_data, x) for x in range(len(csv_data))]
		silhouette = {}
		for p_idx in range(len(csv_data)):
			for cl_idx in range(len(kmeans.best_matches)):
				if p_idx in kmeans.best_matches[cl_idx]:
					if cl_idx not in silhouette.keys():
						silhouette[cl_idx] = [silhouettes_of_all_points[p_idx]]
					else:
						silhouette[cl_idx].append(silhouettes_of_all_points[p_idx])
			silhouettes[valor_k] = silhouette

		wss.append(kmeans.intra_cluster_variation(csv_data))

	# print(silhouettes)

	# Alla on s'aplana la curva es el optim
	create_chart(k, wss)

	# Crear tots els grafics de silhouettes per cada k
	avgs = create_silhouette_chart(k, silhouettes, chart=True)

	# Crear un grafic de les average silhouettes, el maxim es el que ens interesse.
	plt.rcdefaults()

	num_ks = [i + 1 for i in range(len(avgs)) if i != 0]
	bar_ch = plt.bar(num_ks, avgs[1:])
	plt.title("Average silhouettes per numero de clusters")

	plt.show()

# apartat_clustering()
