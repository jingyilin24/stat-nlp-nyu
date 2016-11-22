'''
Created on Nov 13, 2016

@author: jylkb
'''
import gensim, logging
from gensim.models.word2vec import Word2Vec
from scipy.stats import spearmanr
import infer
from sets import Set

# a sentences reading class that take inputs one at a time without loading all data in RAM
class Sentences(object):
    def __init__(self, file_name, count):
        self.file_name = file_name
        self.count = count

    def __iter__(self):
        count = 0
        for line in open(self.file_name):
            if count > self.count:
                break
            else:
                count += 1
                yield line.lower().split(' ')

# pairs class for wordSim pairs
class Pair(object):
    def __init__(self, first, second, score):
        self.first = first
        self.second = second
        self.score = score   

# read in the wordSim pairs to be the evaluation set
def read_wordSimPairs( wordSim_dir ):
    pairs = []
    file = open( wordSim_dir)
    # the first line is the header
    header = True
    for line in file:
        if header:
            header = False
        else:
            pair = line.split(',')
            p = Pair( pair[0], pair[1], pair[2] )
            pairs.append(p);
    file.close()    
    return pairs
        
# calculate the spearmans score
def spearmansScore(wordSim_pairs, model, dim):
    predictions = []
    labels = []
    
    for pair in wordSim_pairs:
        first_word = pair.first
        second_word = pair.second
        labels.append( pair.score )
        if first_word in model.vocab and second_word in model.vocab:
            predictions.append( model.similarity(first_word, second_word) )
        else:
            predictions.append( 0.5 )
    return spearmanr(predictions, labels)[0]

def output_embeddings(wordSim_pairs, model, dim):
    #write the embeddings on the a txt file
    embeddings = open('C:\Users\jylkb\Dropbox\Math_in_Finance\FA16\NLP\HWs\HW5\embeddings.txt', 'w')
    num_words = get_number_of_words(wordSim_pairs)
    embeddings.write( '%d %d\n' % (num_words, dim) )
    words_written = Set()
    
    for pair in wordSim_pairs:
        first_word = pair.first
        first_word_str_vec = get_str_vec( first_word, model, dim )
        second_word = pair.second
        second_word_str_vec = get_str_vec( second_word, model, dim )
        # write the word and embeddings into the file
        if first_word not in words_written:
            embeddings.write( '%s %s\n' % (first_word, first_word_str_vec) )
        if second_word not in words_written:
            embeddings.write( '%s %s\n' % (second_word, second_word_str_vec) )
        words_written.add( first_word )
        words_written.add( second_word )

def get_number_of_words(wordSim_pairs):
    words = Set()
    for pair in wordSim_pairs:
        first_word = pair.first
        second_word = pair.second        
        words.add(first_word)
        words.add(second_word)
    return len(words)
            
def get_str_vec(word, model, dim):
    str_vec = ''
    if word in model.vocab:
        vec = model[word] # numpy array
        for i in range(dim-1):
            str_vec += str(vec[i]) + ' '
        str_vec += str(vec[dim-1])
    else:
        vec = [0] * dim
        for i in range(dim-1):
            str_vec += str(vec[i]) + ' '
        str_vec += str(vec[dim-1])
    return str_vec
        
def word2vec(dim, window, min_count):
    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    data_dir = 'C:\\Users\\jylkb\\Dropbox\\Math_in_Finance\\FA16/NLP\\source_code_data\\DATA\\data5\\'
    training_dir = data_dir + 'training-data\\training-data.1m'
    wordSim_dir = data_dir + 'wordsim353\\combined.csv'
    num_training_data = int(1e10)
    training_sentences = Sentences(training_dir, num_training_data)
    wordSim_pairs = read_wordSimPairs( wordSim_dir )
    model = Word2Vec(training_sentences, size=dim, window=window, min_count=min_count, workers=10)
    score = spearmansScore( wordSim_pairs, model, dim )
    output_embeddings(wordSim_pairs, model, dim)
    print 'Spearman rank correlation: %f' % score 
    

def word2vecf():
    root_dir = 'C:\\Users\\jylkb\\Google Drive\\word2vecf\\'
    npy_dir = root_dir + 'vecs.npy'
    vocab_dir = root_dir + 'vecs.vocab'
    wordSim_dir = 'C:\\Users\\jylkb\\Dropbox\\Math_in_Finance\\FA16/NLP\\source_code_data\\DATA\\data5\\' + 'wordsim353\\combined.csv'
    wordSim_pairs = read_wordSimPairs( wordSim_dir )
    e = infer.Embeddings(npy_dir, vocab_dir)
    score = spearmansScore( wordSim_pairs, e )
    print 'Spearman rank correlation: %f' % score

def word2vecOptiomize():
    dims = [50, 100, 150, 200]
    windows = [5, 10, 15, 20, 30]
    min_counts = [5, 10, 20, 50, 100]
    score_dict = {}
    max_score = 0
    max_param = ''
    for dim in dims:
        for window in windows:
            for min_count in min_counts:
                score = word2vec(dim, window, min_count)
                param = 'Dim: %d, Window: %d, Min_count: %d' % (dim, window, min_count)
                print param
                print score
                if score > max_score:
                    max_score = score
                    max_param = param
                score_dict[param] = score
    print max_score
    print max_param
    
if __name__ == '__main__':
    word2vec(1000, 30, 10)

























