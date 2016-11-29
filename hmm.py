
#part 2
#emission param
#storing position word position i with tag i

words_x = []
tags_y = []

def populate_word_tag_lists(training_set):
    training_set = readTraining(xyz)
    for i in range(0,len(training_set)):
        for j in range(0,len(training_set[i])):
            words_x.add(training_set[i][j][0])
            tags_y.add(training_set[i][j][1])        



def count_emission_y_to_x(x,y):
	count_yx = 0
	for i in range (0, len(tags_y)):
		if words_x[i] == y:
			if tags_y[i] == x:
				count += 1
    return count_yx

def count_emission_y(y):
    count_y = 0
    for i in range (0, len(tags_y)):
        if tags_y == y:
            count_y += 1
    return count_y


def emission_param(x,y):
    if x not in words_x:
        #new word
        count_yx = 1
    else:
        #not new
        count_yx = count_emission_y_to_x(x,y)
    count_y = count_emission_y(y) + 1
    return count_emission_y_to_x / count_emission_y


def get_optimal_y(x):
    emission_param_set = {}
    for y in tags_y:
        emission_param_set[y] = emission_param(x,y)
    optimal_y = max(emission_param_set, key=emission_param_set.get)
    return optimal_y


#read training/testing data
def readTraining(filename):
    training_set = []
    file = open(filename, 'r').read()
    for i in file.split("\n\n"):
        tweet = []
        for j in i.split("\n"):
            if len(j) == 0:
                continue
            tweet.add(j.split("\t"))
        training_set.add(tweet)
    return training_set





