def filter_token(token):
    try:
        return int(token)
    except ValueError:
        return token


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
            print(line)
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
# Value: Pot ser numero o no, serà el valor que comprovarem al index column, si es numero mirarem si el registre és més gran, sino mirarem si és igual.
def divide_set(part, column, value):
    split_function = None
	
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
	# value: Solament a nodes, és el valor que satisfà la pregunta i separar els registres (Si és número, la pregunta és >= value)
	# results: Solament a fulles, és un diccionari que guarda els resultats d'aquesta branca, haurien de ser de sol 1 classe per ser ideal
	# tb i fb: Solament a nodes, son els nodes resultants de partir segons la pregunta feta amb value a col
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb

def buildtree(part, scoref=entropy, beta=0):
	if len(part) == 0: return decisionnode(results=[])
	current_score = scoref(part) #i(t)
	
	best_gain = 0 #max inc i(s, t)   (Com mes gran millor)
	best_column = None
	best_value = None
	best_sets = None
	
	# Per tots els atributs que tinguin alguna diferencia (no te sentit partir per algo que son tots iguals) menys el ultim
		# Buscar un value per on partir (si es numero fer alguna mitja o algo, si es string ferho per tots els que apareixen
			# partir amb divide_set(part, columna que sigui, value que sigui)
			# calcular el gain: i(t) - pl*i(tl) - pr*i(tr) on t es part, tl es set1, tr es set2, pl es len(set1)/len(part) i pr es len(set2)/len(part)
			# Si es millor que el best_gain canviar el que toqui
	
	if best_gain >= beta: # Continuem la construcció
		# Retornar un decisionnode cridant a la funcio a tb i fb amb set1, set2 i les mateixes scoref i beta
		pass
	else:
		# El node es terminal
		return decisionnode(results=unique_counts(part))
		

readfile = read_file("decision_tree_example.txt", ignore_first_line=True)
print(unique_counts(readfile))
print(gini_impurity(readfile))
print(entropy(readfile))
print(divide_set(readfile, 0, "google"))
