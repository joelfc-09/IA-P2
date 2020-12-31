from functools import reduce
from DecisionTrees import printtree


def filter_token(token):
	try:
		return int(token)
	except ValueError:
		return token

def to_register(string):
	return list(map(filter_token,string.split(",")))

def read_file(file_path, data_sep=",", ignore_first_line=False):
	prototypes = []

	with open(file_path) as f:
		# Strip lines
		strip_reader = (l.strip() for l in f)  # Al fer-ho directament sobre f es fa lazy
		# Filtrar lineas vacias
		filtered_reader = (l for l in strip_reader if l)
		# eliminar primera linea si necessari
		if ignore_first_line:
			next(filtered_reader)
		# tokenitzar i afegir a prototypes
		for line in filtered_reader:  # També serveix per a consumir el generador
			# print(line)
			prototypes.append([filter_token(token) for token in line.split(data_sep)])
	return prototypes


# Retorna quants elements de cada classe tenim segons l'atribut objectiu (assumim que és l'ultim)
def unique_counts(part):
	results = {}
	for pr in part:
		results[pr[-1]] = results.get(pr[-1], 0) + 1
	return results


# Impuresa de gini
def gini_impurity(part):
	total = float(len(part))
	results = unique_counts(part)
	return 1 - sum((v / total) ** 2 for v in results.values())


# Entropia
def entropy(part):
	import math
	total = float(len(part))
	results = unique_counts(part)
	return - sum((v / total) * math.log(v / total, 2) for v in results.values())


# Divideix el set part en 2 sets comparant amb value a la columna amb index column, retorna una còpia
# Part: Llista de llistes, son els registres amb tots els atributs
# Column: Index dels registres que compararem amb value
# Value: Pot ser numero o no, serà el valor que comprovarem al index column,
# si es numero mirarem si el registre és més gran, sino mirarem si és igual.
def divide_set(part, column, value):
	# Segons si és numero o no
	if isinstance(value, int) or isinstance(value, float):
		split_function = lambda row: row[column] >= value
	else:
		split_function = lambda row: row[column] == value

	set1 = []
	set2 = []
	for elem in part:
		if split_function(elem):
			set1.append(elem)
		else:
			set2.append(elem)
	# Set1 compleix la condició, set2 no
	return set1, set2


class decisionnode:
	# col: Solament a nodes, es l'index de la columna que comprovem per fer la pregunta i separar els registres
	# value: Solament a nodes, és el valor que satisfà la pregunta per separar els registres
	# (Si és número, la pregunta és >= value)
	# results: Solament a fulles, és un diccionari que guarda els resultats d'aquesta branca,
	# haurien de ser de sol 1 classe per ser ideal
	# tb i fb: Solament a nodes, son els nodes resultants de partir segons la pregunta feta amb value a col
	def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
		self.col = col
		self.value = value
		self.results = results
		self.tb = tb
		self.fb = fb

	def print(self):
		if self.results is None:
			print("Col: ", self.col, "Value: ", self.value)
		else:
			print(self.results)


def calculate_gain(current_score, set1, set2, scoref=entropy):
	total_len = len(set1) + len(set2)
	set1_weighted = scoref(set1) * (len(set1) / total_len)
	set2_weighted = scoref(set2) * (len(set2) / total_len)
	return current_score - set1_weighted - set2_weighted


def find_best_partition(part, scoref=entropy, numeric_step=0.1):
	current_score = scoref(part)  # i(t)

	best_gain = 0  # max inc i(s, t)   (Com mes gran millor)
	best_column = None
	best_value = None
	best_sets = None
	for column_index in range(len(part[0]) - 1):  # Sabem que part té almenys un registre
		# Per tots els atributs que tinguin alguna diferencia menys el ultim
		if not reduce((lambda x, y: x == y), [registre[column_index] for registre in part]):
			# Buscar un value per on partir
			if isinstance(part[0][column_index], str):  # Si es string fer-ho per tots
				present_values = list(set([registre[column_index] for registre in part]))
				for present_value in present_values:
					# Partir i mirar si té una millor puntuació
					set1, set2 = divide_set(part, column_index, present_value)
					gain = calculate_gain(current_score, set1, set2, scoref)
					if gain > best_gain:
						best_gain = gain
						best_column = column_index
						best_value = present_value
						best_sets = set1, set2
			else:  # Es numeric
				min_value = reduce(min, [registre[column_index] for registre in part])
				max_value = reduce(max, [registre[column_index] for registre in part])
				numeric_value = min_value
				while numeric_value < max_value:
					set1, set2 = divide_set(part, column_index, numeric_value)
					gain = calculate_gain(current_score, set1, set2, scoref)
					if gain > best_gain:
						best_gain = gain
						best_column = column_index
						best_value = numeric_value
						best_sets = set1, set2
					numeric_value += numeric_step
	return best_gain, best_column, best_value, best_sets


def buildtree(part, scoref=entropy, beta=0.0, numeric_step=0.1):
	if len(part) == 0:
		return decisionnode(results=[])
	best_gain, best_column, best_value, best_sets = find_best_partition(part, scoref=scoref,numeric_step=numeric_step)

	if best_gain >= beta:  # Continuem la construcció
		tb = buildtree(best_sets[0], scoref=scoref, beta=beta, numeric_step=numeric_step)
		fb = buildtree(best_sets[1], scoref=scoref, beta=beta, numeric_step=numeric_step)
		return decisionnode(best_column, best_value, None, tb=tb, fb=fb)
	else:
		# El node es terminal
		return decisionnode(results=unique_counts(part))


def buildtree_iterative(part, scoref=entropy, beta=0.0, numeric_step=0.1):
	if len(part) == 0:
		return decisionnode(results=[])
	root_node = decisionnode(results=part)
	nodes = [root_node]
	while len(nodes) > 0:
		node = nodes.pop(0)
		partition = node.results
		best_gain, best_column, best_value, best_sets = find_best_partition(partition, scoref=scoref,
		                                                                    numeric_step=numeric_step)
		if best_gain >= beta:
			node.col = best_column
			node.value = best_value
			node.tb = decisionnode(results=best_sets[0])
			node.fb = decisionnode(results=best_sets[1])
			nodes.append(node.tb)
			nodes.append(node.fb)
			node.results = None
		else:
			node.results = unique_counts(partition)

	return root_node


def classify(root_node, register):
	print("Classifying register... ", register)
	current_node = root_node
	current_node.print()
	while current_node.results is None:
		if isinstance(current_node.value, str):
			if current_node.value == register[current_node.col]:
				current_node = current_node.tb
			else:
				current_node = current_node.fb
		else:
			if current_node.value < register[current_node.col]:
				current_node = current_node.tb
			else:
				current_node = current_node.fb
	if len(current_node.results) == 1:
		print("Register is most likely ", list(current_node.results.keys()))
	else:
		print("Register is between ", " or ".join([str(k) for k in current_node.results.keys()]))

readfile = read_file("decision_tree_extended.txt", ignore_first_line=True)

nodetest = buildtree(readfile, scoref=gini_impurity, beta=0.001)
printtree.printtree(nodetest)

classify(nodetest,to_register("araknet,France,no,29,gencoin"))
