
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
        self.verbose = verbose  # False, True or 'extreme' (for debugging)
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

        self.problem_tweets = []

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
                    self.tag_count_dict[tag])

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
        if y in ['__STOP__', '__START']:
            return 0 # these don't emit words
        try:
            count_yx = self.emission_dict[y][x]
        except:
            # return 1.0 / len(self.word_set)
            if x not in self.word_set:  # new word that wasn't in training set
                return 1.0 / len(self.word_set) * self.tag_count_dict[y]*1.0/ len(self.words_x)
                # count_yx = 1
            else: # word that wasn't emitted by this tag before
                return 0
                # return 0.001/ len(self.words_x) #* self.tag_count_dict[y] * 1.0/ len(self.words_x)


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
                word = self.classify_word(word)

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
                word = self.classify_word(word)
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

        # counts['__STOP__'] = counts['__START__']
        # self.tag_counts = counts

        for tag in self.transition_dict:
            tag_count = self.tag_count_dict[tag]
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
            score, pairs = self.viterbi(tweet)
            # except:
            #     pairs = [(word, 'O') for word in tweet] # just for now okay

            for word, tag in pairs:
                fo.write(word)
                fo.write(' ')
                fo.write(tag)
                fo.write('\r\n')
            fo.write('\r\n')
        fo.close()

    def viterbi_helper_old(self, sentence, k):
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

    def viterbi_helper(self, sentence):
        # precomputes the pi function
        # recursively gets pi values for
        # all indices from 1 to k
        # and all tags v
        # and returns them in a big lump
        # avoids having to calculate same values over and over again

        tags = self.tag_set + ['__START__', '__STOP__']
        index = 0
        # base case
        # construct base dict
        pi_table = [{}]
        for tag in tags: # initial when k = 0
            pi_table[0][tag] = 1 if (tag == '__START__') else 0

        for k in range(len(sentence)): # goes from 0 to n - 1
            d = {}
            prev_pi = pi_table[k] #[-1]
            word_k = sentence[k]  # word at position k, cos k starts from 1
            all_empty = True
            if self.verbose == 'extreme':
                print 'Current k:   \t', k
                print 'Current word:\t', word_k
                print '-------- Partial pi table ---------'
                self.pretty_print_pi(pi_table, sentence)


            for v in tags:
                max_score = 0
                for u in tags:
                    score = prev_pi[u] * \
                        self.transition_param(u, v) * \
                        self.emission_param_fixed(word_k, v)
                    if score > 0:
                        all_empty =  False
                        if self.verbose == 'extreme':
                            print 'transition score', u, '->', v, self.transition_param(u, v)
                            print 'emission score', v, '->', word_k, self.emission_param_fixed(word_k, v)
                            print 'previous pi score:', u, prev_pi[u]
                            print 'total score', score
                            print '\n'
                    if score > max_score:
                        max_score = score
                        # if self.verbose == 'extreme':
                        #     print 'new score for {}: {} > {}'.format(v, score, max_score)
                        #     print '\n'
                d[v] = max_score
            if all_empty:
                if self.verbose:
                    print 'all empty...', '#\t'+' '.join(sentence)
                tag, = self.get_max_key(prev_pi)
                # emission_term_estimate =
                d['O'] = self.transition_param(tag,'O') * 1.0/len(self.emission_dict['O']) * prev_pi[tag]
                if sentence not in self.problem_tweets:
                    self.problem_tweets += [sentence]
            pi_table += [d]
        # self.pretty_print_pi(pi_table, sentence)
        # print '\n'
        return pi_table

    def get_max_key(self, dict_of_values, num = 1):
        max_key = None
        sort_list = list(dict_of_values.iteritems())
        sort_list.sort(key = lambda a:-a[1])
        return tuple([v[0] for v in sort_list[:num]])
        # for key, value in dict_of_values.iteritems():
        #     if not max_key:
        #         max_key = key
        #         max_value = value
        #     elif value > max_value:
        #         max_key = key
        #         max_value = value
        # return max_key






    def viterbi(self, sentence):
        # implements the recursive Viterbi algorithm
        # to return the most probable tags that generated a given sentence
        # k is the word index from 1-n
        # v is the tag

        if not self.emission_dict or not self.transition_dict:
            raise Exception('dictionaries not generated')

        n = len(sentence)
        pi_table = self.viterbi_helper(sentence)

        tags = self.tag_set + ['__START__', '__STOP__']
        sentence_pairs = [''] * n
        next_tag = '__STOP__' # backtrack from last tag
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
                    if i == 0:
                        total_score = score

            # print word, index, tag, max_score, score

            next_tag = tag
            sentence_pairs[index - 1] = (word, tag)

        if self.verbose == 'debug':
            print '-------- Sentence ---------'
            print ' '.join(sentence)
            print '-------- Pi table ---------'
            self.pretty_print_pi(pi_table, sentence)
            print 'Total Score: ', total_score
            print '-------- Tagged sentence ---------'
            self.pretty_print_tagged_sentence(sentence_pairs)
            print '----------------------------------'


        # return max_score, pi_table, sentence_pairs
        return total_score, sentence_pairs

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
                print '{}: {}\t'.format(self.short[pair[0]], self.print_log(pair[1])),

            print sentence[i]

    def pretty_print_tagged_sentence(self,sentence, gray = True):
        # for word, tag in sentence:
        #     print a.short[tag], '\t', word
        # for word, tag in sentence:
        print '#\t', ''.join([a.short[tag]+' '*(len(word)-1) for word, tag in sentence])

        print '#\t', ' '.join([word for word, tag in sentence])

    def print_log(self,num):
        if num == 0:
            return '_____'
        else:
            return '{:05.1f}'.format(math.log(num,10))

    def classify_word(self,word):
        # if word[0] == '#':
        #     return "__HASHTAG__"
        # if word[0] == '@':
        #     return "__USERNAME__"
        if word[0] in "#@":
            return word.lower() # twitter usernames and hashtags are case insensitive
        if 'http://' in word:
            # print word
            pass
            # return "__{}__".format(word.split('/')[2]) # strip to domain name
        # else:
        return word

    def top(self, k):
        sentence_open = [([('__START__', 'None')], 1)]
        k_open = k
        # sentence_open = sentences
        # return sentences
        tags = self.tag_set# + ['__STOP__']
        result = []
        while k_open > 0:
            # next_sentence = sentence_open # tuples of sentence list, score
            next_sentence = []
            for sentence, prev_pi in sentence_open:
                prev_tag = sentence[-1][0]
                for tag in tags:
                    # num = k_open

                    best_words = self.get_max_key(self.emission_dict[tag], k_open)
                    # print tag, best_words
                    for word in best_words:
                        sentence_candidate = sentence+[(tag, word)]
                        sentence_pi = prev_pi * self.transition_param(prev_tag, tag) * self.emission_param_fixed(word, tag)
                        next_sentence += [(sentence_candidate, sentence_pi)]
                stop_pi = prev_pi*self.transition_param(prev_tag, '__STOP__')
                next_sentence += [(sentence+[('__STOP__', None)], stop_pi)]
            # next_sentence += sentence_open # include existing sentences
            # print next_sentence
            # time.sleep(2)
            next_sentence.sort(key= lambda a:-a[1])
            next_sentence = next_sentence[:k_open] # trim to k-open top best sentences
            sentence_open = []
            for sentence, score in next_sentence:
                if sentence[-1][0] == '__STOP__':
                    result += [(sentence, score)]
                    k_open -= 1
                else:
                    sentence_open += [(sentence, score)]
            # print len(sentence_open)
            # for sentence, score in sentence_open:
            #     print score,'\t', ' '.join([tup[1] for tup in sentence])
            # time.sleep(1)
        return result













which_part = [3, 4]


a = HMM(directory, "EN")
a.populate_word_tag_lists()
if 2 in which_part:
    devin = a.readDevIn()
    a.create_optimal_dict(devin)
    a.writePredictionP2()


if 3 in which_part:
    a.readDevIn()
    a.create_emission_dict()
    a.create_transition_dict()

    a.writePredictionP3()

if 4 in which_part:
    best_tweets = a.top(10)
    best_tweets = [[word for tag, word in tweet] for tweet,score in best_tweets]
    best_tweets = [' '.join(tweet[1:-1]) for tweet in best_tweets]
    print '\n'.join(best_tweets)



if 'debug' in which_part:

    def has_link(tweet):
        for word in tweet:
            if '__' in word:
                return True
        return False
    print '#\t' + '\n#\t'.join([' '.join(tweet) for tweet in a.dev_tweets if has_link(tweet)])

    a.verbose = 'debug'
    tweet = a.dev_tweets[4]
    # tweet = problem_tweets[3]
    score, pairs = a.viterbi(tweet)

    for tweet in a.dev_tweets[:10]:
        a.viterbi(tweet)



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
