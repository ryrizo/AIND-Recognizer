import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        self.p = next(iter(all_word_Xlengths.values()))[0].shape[1]
        self.N = next(iter(all_word_Xlengths.values()))[0].shape[0]

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """
    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    where L is the likelihood of the fitted model, p is the number of parameters,
and N is the number of data points.
    """

    def select(self, verbose=False):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Fencepost
        best_num_states = self.min_n_components
        best_score = self.bic_score(self.min_n_components)
        scores = set()
        scores.add((best_num_states, best_score))
        for i in range(self.min_n_components+1,self.max_n_components+1):
            curr_score = self.bic_score(i)
            if verbose:
                print("Word: {} Num States: {} BIC: {}".format(self.this_word,i,curr_score))
            if curr_score == None:
                continue
            scores.add((i,curr_score))

        if best_score is None or scores is None:
            return self.base_model(best_num_states)
        else:
            for states, score in scores:
                if score > best_score:
                    best_num_states = states
                    best_score = score
        return self.base_model(best_num_states)

    def bic_score(self, num_states, verbose=False):
        try:
            temp_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            logL = temp_model.score(self.X, self.lengths)
            bic_score = -2 * logL + self.p * np.log(self.N)
            return bic_score
        except ValueError as e:
            if verbose:
                print(e)
            return None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    logL - SUM(logL every other word)/M-1
    '''

    def select(self, verbose=False):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Fencepost
        best_num_states = self.min_n_components
        best_score = self.dic_score(self.min_n_components)
        scores = set()
        scores.add((best_num_states, best_score))
        for i in range(self.min_n_components+1,self.max_n_components+1):
            curr_score = self.dic_score(i)
            if verbose:
                print("Word: {} Num States: {} DIC: {}".format(self.this_word,i,curr_score))
            if curr_score == None:
                continue
            scores.add((i,curr_score))

        if best_score is None or scores is None:
            return self.base_model(best_num_states)
        else:
            for states, score in scores:
                if score > best_score:
                    best_num_states = states
                    best_score = score
        return self.base_model(best_num_states)

    def dic_score(self, num_states, verbose=False):
        try:
            temp_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            logL = temp_model.score(self.X, self.lengths)
            other_logL = 0
            for word in self.hwords:
                if word == self.this_word:
                    continue
                X_temp, length_temp = self.hwords[word]
                other_logL += temp_model.score(X_temp, length_temp)
            M = len(self.hwords.keys())
            dic_score = logL - other_logL/(M-1)
            return dic_score
        except ValueError as e:
            if verbose:
                print(e)
            return None



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
        # Need try catch to eliminate some models
    '''
    def select(self, verbose=False):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Fencepost
        best_num_states = self.min_n_components
        best_score = self.kfold_score(self.min_n_components)
        for i in range(self.min_n_components+1,self.max_n_components+1):
            curr_score = self.kfold_score(i)
            if verbose:
                print("Word: {} Num States: {} LogL: {}".format(self.this_word,i,curr_score))
            if curr_score == None:
                return self.base_model(self.min_n_components) # Cannot select a model
            if curr_score > best_score:
                best_num_states = i
                best_score = curr_score
        return self.base_model(best_num_states)

    def kfold_score(self, num_states, verbose=False):
        scores = 0
        split_method = KFold()
        try:
            for train_idx, test_idx in split_method.split(self.sequences):
                X_train, lengths_train = combine_sequences(train_idx,self.sequences)
                X_test, lengths_test = combine_sequences(test_idx,self.sequences)
                temp_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                scores += temp_model.score(X_test,lengths_test)
            kfold_score = scores/split_method.n_splits
            return kfold_score
        except ValueError as e:
            if verbose:
                print(e)
            return None
