import random

random.seed(None)

referrers = ["google", "slashdot", "digg", "kiwitobes", "(direct)", "youtube", "araknet", "guizer"]
countries = ["USA", "UK", "France", "Spain", "NewZealand", "Mexico", "Russia"]
yesno = ["yes","no"]
pages_min = 12
pages_max = 35
services = ["None", "Basic", "Premium"]
payments = ["bank_transfer","gencoin","paypal","bitcoin","prepaid"]

def generate_row():
	referrer = referrers[random.randint(0,len(referrers) - 1)]
	country = countries[random.randint(0, len(countries) - 1)]
	faq = yesno[random.randint(0, 1)]
	pages = random.randint(pages_min,pages_max)
	payment = payments[random.randint(0, len(payments) - 1)]
	service = services[random.randint(0, len(services) - 1)]
	return ",".join([referrer,country,faq,str(pages),payment,service])

with open("decision_tree_extended.txt","a") as f:
	for i in range(100):
		f.write(generate_row() + "\n")
