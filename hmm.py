
#part 2
#emission param
#storing position word position i with tag i



def populate_word_tag_lists(filename):
    words_x = []
    tags_y = []
    training_set = readTraining(filename)
    for i in training_set:
        if len(i) == 2:
            words_x.append(i[0])
            tags_y.append(i[1])
    return words_x, tags_y

def count_emission_y_to_x(x,y):
    count_yx = 0
    for i in range(0, len(tags_y)):
        if words_x[i] == x:
            if tags_y[i] == y:
                count_yx += 1
    return count_yx

def count_emission_y(y):
    count_y = 0
    for i in tags_y:
        if i == y:
            count_y += 1
    return count_y


def emission_param(x,y):
    count_yx = count_emission_y_to_x(x,y)
    count_y = count_emission_y(y)
    return count_yx / count_y

def emission_param_fixed(x,y):
    if x not in words_x:
        count_yx = 1
    else:
        count_yx = count_emission_y_to_x(x,y)
    count_y = count_emission_y(y) + 1
    return count_yx / count_y

def create_emission_dict(dev_in_word):
    emission_dict = {}
    for i in dev_in_word:
        emission_dict[i] = get_optimal_y(i)
    return emission_dict

def get_optimal_y(x):
    emission_param_set = {}
    for y in tags_y:
        emission_param_set[y] = emission_param_fixed(x,y)
    optimal_y = max(emission_param_set, key=emission_param_set.get)
    return optimal_y


#read training/testing data
def readTraining(filename):
    training_set = []
    fo = open(filename, 'r')
    while True:
        line = fo.readline()
        if line == '\n':
            continue
        elif line == '':
            break
        else:
            training_set.append(line.replace('\r\n','').split(' '))
    fo.close()
    return training_set

def readDevIn(filename):
    dev_in_word = []
    fo = open(filename, 'r')
    while True:
        line = fo.readline()
        if line == '\n':
            continue
        elif line == '':
            break
        else:
            dev_in_word.append(line.replace('\r\n',''))
    fo.close()
    return dev_in_word

# def make_prediction(training, dev_in):
#     prediction = {}
#     for i in dev_in:
#         prediction[i] = training[i]
#     return prediction

def writePrediction(filename, result):
    fo = open(filename, 'w')
    for key in result:
        fo.write(key)
        fo.write(' ')
        fo.write(result[key])
        fo.write('\n')
    fo.close()

words_x, tags_y = populate_word_tag_lists('/home/tuananh/Documents/ML/Project/ml-project-2016/EN/EN/train')
dev_in_word = readDevIn('/home/tuananh/Documents/ML/Project/ml-project-2016/EN/EN/dev.in')
emission_dict = create_emission_dict(dev_in_word)
# prediction_dict = make_prediction(emission_dict,dev_in_word)
# writePrediction('dev.p2.out',prediction_dict)
print len(words_x), len(tags_y)
print emission_dict
