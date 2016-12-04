
# part 2
# emission param
# storing position word position i with tag i
import time
import math

who_am_i = 'yuhan'
# who_am_i = 'anh'
# who_am_i = 'anh2'

directory = {'anh2': 'D:/Project/ml-project-2016',
             'anh': '/home/tuananh/Documents/ML/Project/ml-project-2016',
             'yuhan': '/Users/yuhan/Documents/SUTD/,Machine Learning/ML_project'}[who_am_i]

language = 'EN'


class HMM(object):
    def __init__(self, directory, language, verbose=True):
        self.parent_dir = directory
        self.directory = directory + '/{0}/{0}'.format(language)
        self.language = language
        self.verbose = verbose  # False, TRUE or 'extreme' (for debugging)
        self.words_x = []
        self.tags_y = []
        self.tag_set = []  # reduce tags_y to minimal set, doesn't include start and stop tags
        self.word_set = set([]) # muchhhh faster way to test for inclusion, O(1) vs O(n) for list
        self.training_set = []  # tuple pairs of word / tag
        self.tag_count_dict = {}
        self.optimal_dict = {}  # previously named emission dict, gives optimal tag for each word
        self.emission_dict = {}  # gives probability of emitting each word for each tag
        # dictionary of dictionaries, access probabilty by [tag][next_tag]
        self.transition_dict = {}
        # self.pair_dict = {}
        self.tweets = []
        self.dev_words = []
        self.dev_tweets = []

        self.short = {'__STOP__':   'S>',
                      'B-neutral':  'B0',
                      'B-negative': 'B-',
                      'O':          'O ',
                      '__START__':  'S<',
                      'B-positive': 'B+',
                      'I-neutral':  'I0',
                      'I-positive': 'I+',
                      'I-negative': 'I-'}

        if verbose:
            print 'Initialized HMM class with language ' + language

    def populate_word_tag_lists(self):
        if self.verbose:
            print "Populating word tag lists ..."
            print 'Creating emission dictionary ...'  # at the same time yo

        training_file = self.directory + '/train'
        self.words_x = []
        self.tags_y = []
        self.readTraining(training_file)  # returns a list of lists
        for pair in self.training_set:
            if len(pair) == 2:  # ignore empty lines
                word, tag = pair  # unpack values
                self.words_x.append(word)
                self.tags_y.append(tag)
                if tag not in self.tag_count_dict:
                    # first time seeing this tag
                    self.tag_count_dict[tag] = 1
                    self.emission_dict[tag] = {word: 1}
                else:
                    self.tag_count_dict[tag] += 1
                    try:
                        self.emission_dict[tag][word] += 1
                    except:
                        # first time seeing word attached to this tag
                        self.emission_dict[tag][word] = 1
            # else:
            #     self.tag_count_dict['__START__'] += 1
            #     self.tag_count_dict['__STOP__'] += 1

        # kinda hacky
        self.tag_count_dict['__START__'] = len(self.tweets)
        self.tag_count_dict['__STOP__'] = len(self.tweets)

        self.tag_set = list(set(self.tags_y))
        self.word_set = set(self.words_x)
        # normalize emission values to 0-1 probability
        for tag in self.emission_dict:
            for word in self.emission_dict[tag]:
                self.emission_dict[tag][word] /= float(
                    self.tag_count_dict[tag] + 1)

    def count_emission_y_to_x(self, x, y):
        count_yx = 0
        test_pair = (x, y)
        # if test_pair in self.pair_dict:
        #     return self.pair_dict[pair]

        for pair in self.training_set:
            if test_pair == pair:
                count_yx += 1
        # self.pair_dict[test_pair] = count_yx
        return count_yx

    def emission_param(self, x, y):
        # count_yx = self.count_emission_y_to_x(x, y)
        count_yx = self.emission_dict[y][x]
        count_y = self.tag_count_dict[y]
        return float(count_yx) / count_y

    def emission_param_fixed(self, x, y):
        # modified algorithm used for test data
        try:
            count_yx = self.emission_dict[y][x]
        except:
            if x not in self.word_set:  # new word that wasn't in training set
                count_yx = 1
            else:
                count_yx = 0

        # try:
        #     count_yx = self.emission_dict[y][x]
        # except KeyError as e:
        #     count_yx = 0
            # count_yx = 1

        count_y = self.tag_count_dict[y] + 1.0
        return float(count_yx) / count_y

    def get_optimal_y(self, x):
        # gets the optimal tag for a given word
        emission_param_set = {}

        # for all the possible tags, assign an emission parameter to the word x
        optimal_y = None
        optimal_param = 0
        for y in set(self.tags_y):
            param = self.emission_param_fixed(x, y)
            if param >= optimal_param:
                optimal_y = y
                optimal_param = param

        if self.verbose == 'extreme':
            print 'Optimal for {}: {}, score {}'.format(x, optimal_y, optimal_param)

        return optimal_y

    def create_optimal_dict(self, dev_in_word=None):
        starttime = time.time()
        if not dev_in_word:
            # do it for every word
            self.optimal_dict = {}
            dev_in_word = self.words_x
            name = "all words"
        else:
            name = "dev.in"

        if self.verbose:
            print "Creating optimal-tag dict for words in {} ... ".format(name),

        for word in dev_in_word:
            if word != '__NEWTWEET__' and word not in self.optimal_dict:  # avoid repeated calculations
                self.optimal_dict[word] = self.get_optimal_y(word)

        if self.verbose:
            print 'done! ({:.2f} s)'.format(time.time() - starttime)

    # read training/testing data

    def readTraining(self, filename):
        if self.verbose:
            print "Reading training file"

        self.training_set = []
        fo = open(filename, 'r')
        tweet = ['__START__']  # temp variable, starts with special start tag

        while True:
            line = fo.readline()
            if line in ['\n', '\r', '\r\n']:  # end of tweet
                tweet.append('__STOP__')
                self.tweets.append(tweet)
                tweet = ['__START__']  # reset tweet

            elif line == '':  # EOF
                break
            else:
                word, tag = line.replace('\r\n', '').split(' ')
                self.training_set.append((word, tag))
                tweet.append(tag)
        fo.close()

    def readDevIn(self):
        filename = self.directory + '/dev.in'
        if self.verbose:
            print 'Reading test file dev.in... '

        tweet = []
        dev_tweets = []
        dev_words = []
        fo = open(filename, 'r')

        while True:
            line = fo.readline()
            if line in ['\n', '\r', '\r\n']:
                dev_words.append('__NEWTWEET__')
                dev_tweets.append(tweet)
                tweet = []  # reset tweet
            elif line == '':  # EOF
                break
            else:
                word = line.replace('\r\n', '')
                dev_words.append(word)
                tweet.append(word)
        fo.close()

        self.dev_words = dev_words
        self.dev_tweets = dev_tweets
        return dev_words

    def writeDict(self):
        filename = self.directory + '/dict.out'
        fo = open(filename, 'w')
        for key, value in self.optimal_dict.iteritems():
            fo.write(key)
            fo.write(' ')
            fo.write(value)
            fo.write('\n')
        fo.close()


#####################################################
########### yu han's code ###########################

    def writePredictionP2(self, dev_in_words=None):
        if not dev_in_words:
            dev_in_words = self.readDevIn()

        if self.verbose:
            print 'Writing output file dev.p2.out ...'
        filename = self.directory + '/dev.p2.out'

        fo = open(filename, 'w')
        for word in dev_in_words:
            if word == '__NEWTWEET__':
                fo.write('\n')
            else:
                fo.write(word)
                fo.write(' ')
                fo.write(self.optimal_dict[word])
                fo.write('\n')
        fo.close()


    def create_transition_dict(self):
        if self.verbose:
            print 'Creating transition dictionary ...'
        self.transition_dict = {}
        counts = {}

        tags = self.tag_set + ['__START__', '__STOP__']
        for tag in tags:
            self.transition_dict[tag] = {}
            counts[tag] = 0
            for next_tag in tags:
                self.transition_dict[tag][next_tag] = 0

        for tweet in self.tweets:
            for i in range(len(tweet) - 1):
                tag = tweet[i]
                next_tag = tweet[i + 1]
                # if tag not in self.transition_dict:
                #     counts[tag] = 1
                #     self.transition_dict[tag] = {}
                # else:
                counts[tag] += 1
                # if next_tag not in self.transition_dict[tag]:
                #     self.transition_dict[tag][next_tag] = 1
                # else:
                self.transition_dict[tag][next_tag] += 1

        counts['__STOP__'] = counts['__START__']
        self.tag_counts = counts

        for tag in self.transition_dict:
            tag_count = counts[tag]
            for next_tag in self.transition_dict[tag]:
                self.transition_dict[tag][next_tag] /= float(tag_count)

    def transition_param(self, from_tag, to_tag):
        return self.transition_dict[from_tag][to_tag]

    def create_emission_dict(self):
        # already created during propagating function
        pass

    def writePredictionP3(self, dev_tweets=None):
        if not dev_tweets:
            if not self.dev_tweets:
                self.readDevIn()
            dev_tweets = self.dev_tweets


        if self.verbose:
            print 'Writing output file dev.p3.out ...'
        filename = self.directory + '/dev.p3.out'

        fo = open(filename, 'w')

        for tweet in dev_tweets:
            try:
                score, pairs = self.viterbi(tweet)
            except:
                pairs = [(word, 'O') for word in tweet] # just for now okay

            for word, tag in pairs:
                fo.write(word)
                fo.write(' ')
                fo.write(tag)
                fo.write('\r\n')
            fo.write('\r\n')
        fo.close()

    def viterbi_helper(self, sentence, k):
        # precomputes the pi function
        # recursively gets pi values for
        # all indices from 1 to k
        # and all tags v
        # and returns them in a big lump
        # avoids having to calculate same values over and over again
        # if k < 0:
        #     # illegal
        #     raise Exception('something is wrong here')

        tags = self.tag_set + ['__START__', '__STOP__']

        d = {}
        if k == 0:
            # base case
            # construct base dict
            for tag in tags:
                d[tag] = 1 if (tag == '__START__') else 0
            return [d]
        else:
            # recursive case oh yeah
            partial_pi_table = self.viterbi_helper(sentence, k - 1)
            prev_pi = partial_pi_table[k - 1] #[-1]
            word_k = sentence[k - 1]  # word at position k, cos k starts from 1
            if self.verbose == 'extreme':
                print 'Current k', k
                print 'Current word', word_k
                print '-------- Partial pi table ---------'
                self.pretty_print_pi(partial_pi_table, sentence)
                # for l in partial_pi_table:
                #     print l
                # print 'previous pi values:'
                # print prev_pi

            for v in tags:
                # print 'next tag: ', v
                max_score = 0
                for u in tags:
                    # print u, v, self.transition_dict[u][v]
                    # print v, word_k, self.emission_dict[u][word_k]
                    # print prev_pi[u]
                    try:
                        score = prev_pi[u] * \
                            self.transition_param(u, v) * \
                            self.emission_param_fixed(word_k, v)
                        # print 'hi'
                        if self.verbose == 'extreme' and score > 0:
                            print 'transition score', u, '->', v, self.transition_param(u, v)
                            print 'emission score', v, '->', word_k, self.emission_param_fixed(word_k, v)
                            print 'previous pi score:', u, prev_pi[u]
                            print 'total score', score
                            print '\n'
                    except KeyError as error:
                        # this should not even happen
                        print 'what error??', v, u, word_k, error
                        score = 0

                    if score > max_score:
                        if self.verbose == 'extreme':
                            print 'new score for {}: {} > {}'.format(v, score, max_score)
                            print '\n'
                        max_score = score
                d[v] = max_score

            return partial_pi_table + [d]

    def viterbi(self, sentence):
        # implements the recursive Viterbi algorithm
        # to return the most probable tags that generated a given sentence
        # k is the word index from 1-n
        # v is the tag

        if not self.emission_dict or not self.transition_dict:
            raise Exception('dictionaries not generated')

        n = len(sentence)
        pi_table = self.viterbi_helper(sentence, n)

        if self.verbose:
            print '-------- Sentence ---------'
            print ' '.join(sentence)
            print '-------- Pi table ---------'
            self.pretty_print_pi(pi_table, sentence)


        tags = self.tag_set + ['__START__', '__STOP__']

        sentence_pairs = [''] * n
        next_tag = '__STOP__'
        total_score = 0

        for i in range(n):
            index = n - i  # this goes backward from n to 1
            word = sentence[index - 1]
            # initialize
            max_score = 0
            tag = ''
            for v in tags:
                score = pi_table[index][v] * self.transition_param(v, next_tag)
                # print v, score
                if score > max_score:
                    tag = v
                    max_score = score
                    if i == 0: total_score = score

            # print word, index, tag, max_score, score

            next_tag = tag
            sentence_pairs[index - 1] = (word, tag)

        if self.verbose:
            print 'Total Score: ', total_score
            print '-------- Tagged sentence ---------'
            a.pretty_print_sentence(sentence_pairs)
            print '----------------------------------'


        # return max_score, pi_table, sentence_pairs
        return total_score, sentence_pairs


# for key, value in a.transition_dict.iteritems():
#     print key, value.keys()

# try:
#     # verify the HMM object exists, otherwise initialise it
#     assert False
#     assert a is not None
# except:

    def pretty_print_pi(self, tab, sentence=None):
        order = {'__START__': 0,
                 'O': 1,
                 'B-negative': 2,
                 'B-neutral': 3,
                 'B-positive': 4,
                 'I-negative': 5,
                 'I-neutral': 6,
                 'I-positive': 7,
                 '__STOP__': 8}
        if not sentence:
            sentence = ['']*len(tab)
        sentence = ['']+sentence
        for i, row in enumerate(tab):
            for pair in sorted(row.iteritems(), key = lambda a:order[a[0]]):
                print '{}: {}\t'.format(self.short[pair[0]], self.log(pair[1])),

            print sentence[i]

    def pretty_print_sentence(self,sentence, gray = True):
        # for word, tag in sentence:
        #     print a.short[tag], '\t', word
        # for word, tag in sentence:
        print '#', ''.join([a.short[tag]+' '*(len(word)-1) for word, tag in sentence])

        print '#', ' '.join([word for word, tag in sentence])

    def log(self,num):
        if num == 0:
            return '_____'
        else:
            return '{:05.1f}'.format(math.log(num,10))




which_part = [3]


a = HMM(directory, "EN")
a.populate_word_tag_lists()
if 2 in which_part:
    devin = a.readDevIn()
    a.create_optimal_dict(devin)
    a.writePredictionP2()


if 3 in which_part:
    a.readDevIn()
    a.create_transition_dict()
    a.verbose = False
    problem_tweets = []
    for tweet in a.dev_tweets:
        try:
            score, pairs = a.viterbi(tweet)
            # a.pretty_print_sentence(pairs)
        except KeyboardInterrupt:
            break
        except:
            print '---ERROR---'  , '\"{}\"'.format(' '.join(tweet))
            problem_tweets += [tweet]

    # a.writePredictionP3()




a.verbose = 'extreme'
# tweet = a.dev_tweets[4]
tweet = problem_tweets[3]
score, pairs = a.viterbi(tweet)


# test_sentence = 'I like the candy'
# pi_list = a.viterbi_helper(test_sentence.split(' '), 4)

# for l in pi_list:
#     print l

# a.readDevIn()

# words_x, tags_y = populate_word_tag_lists(directory + '/EN/EN/train')
# dev_in_word = readDevIn(directory + '/EN/EN/dev.in')
# optimal_dict = create_emission_dict(dev_in_word)

# writePrediction(directory + '/dev.p2.out', optimal_dict)
# # print dev_in_word
