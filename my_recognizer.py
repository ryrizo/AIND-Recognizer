import warnings
from asl_data import SinglesData
import math

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for item in range(0,test_set.num_items):
        word_probabilities_for_item = dict()
        x_item, length_item = test_set.get_item_Xlengths(item)
        for word, model in models.items():
            word_probabilities_for_item[word] = score_model(model,x_item,length_item)
        probabilities.append(word_probabilities_for_item)

    for prob_dict in probabilities:
        guesses.append([(k,v) for k,v in prob_dict.items() if v==max(prob_dict.values())][0][0])

    return probabilities, guesses

def score_model(model, x, lengths,verbose=False):
    try:
        return model.score(x, lengths)
    except Exception as e:
        if verbose:
            print("Exception {}".format(e))
        return -math.inf
