import math
from collections import Counter, defaultdict
from typing import List
from collections import OrderedDict
import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm


class Ngram:
    def __init__(self, config, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config

    def tokenize(self, sentence):
        '''
        E.g.,
            sentence: 'Here dog.'
            tokenized sentence: ['Here', 'dog', '.']
        '''
        return self.tokenizer.tokenize(sentence)

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        '''
        Compute the co-occurrence of each pair.
        '''
        # begin your code (Part 1)
        bigramcnt={}
        unigramcnt={}
        bigrams=[]
        unigrams=[]
        total=0
        for review in corpus_tokenize:
            for idx in range(len(review)-1):
                total+=1
                bigram=(review[idx],review[idx+1])
                unigrams.append(review[idx])
                bigrams.append(bigram)
                
                if review[idx] not in unigramcnt:
                    unigramcnt[review[idx]]=1
                else:
                    unigramcnt[review[idx]]+=1
                    
                if bigram not in bigramcnt:
                    bigramcnt[bigram]=1
                else:
                    bigramcnt[bigram]+=1
        model={}
        self.V=len(unigramcnt.keys())
        self.tt=total
        for bigram in bigrams:
                x=bigram[0]
                y=bigram[1]
                
                prob=bigramcnt[bigram]/unigramcnt[x]
                if x in model:
                    model[x][y]=prob
                else:
                    model[x]=dict()
                    model[x][y]=prob
                    
        features=OrderedDict(sorted(bigramcnt.items(),key=lambda item: -item[1]))
             
        self.unigramcnt=unigramcnt
        self.bigramcnt=bigramcnt
        return model,features
        # end your code
    
    def train(self, df):
        '''
        Train n-gram model.
        '''
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]     # [CLS] represents start of sequence
        
        # You may need to change the outputs, but you need to keep self.model at least.
        self.model, self.features = self.get_ngram(corpus)

    def compute_perplexity(self, df_test) -> float:
        '''
        Compute the perplexity of n-gram model.
        Perplexity = 2^(-entropy)
        '''
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        
        # begin your code (Part 2)
        su=0
        tot=0
        for review in corpus:
            for idx in range(len(review)-1):
                x=review[idx]
                y=review[idx+1]
                feature=(x,y)
                
                if feature in self.features:
                    cnt=self.features[feature]
                else:
                    cnt=0
                numer=(self.bigramcnt[feature]+1 if feature in self.bigramcnt else 1)
                domin=(self.unigramcnt[x] if x in self.unigramcnt else 0)+self.V
                if domin==0:
                    continue
                su+=math.log2(numer/domin)
                tot+=1
        su/=tot
        perplexity=pow(2,-su)                
        # end your code

        return perplexity

    def train_sentiment(self, df_train, df_test):
        '''
        Use the most n patterns as features for training Naive Bayes.
        It is optional to follow the hint we provided, but need to name as the same.

        Parameters:
            train_corpus_embedding: array-like of shape (n_samples_train, n_features)
            test_corpus_embedding: array-like of shape (n_samples_train, n_features)
        
        E.g.,
            Assume the features are [(I saw), (saw a), (an apple)],
            the embedding of the tokenized sentence ['[CLS]', 'I', 'saw', 'a', 'saw', 'saw', 'a', 'saw', '.'] will be
            [1, 2, 0]
            since the bi-gram of the sentence contains
            [([CLS] I), (I saw), (saw a), (a saw), (saw saw), (saw a), (a saw), (saw .)]
            The number of (I saw) is 1, the number of (saw a) is 2, and the number of (an apple) is 0.
        '''
        # begin your code (Part 3)

        # step 1. select the most feature_num patterns as features, you can adjust feature_num for better score!
        feature_num = 1500

        # step 2. convert each sentence in both training data and testing data to embedding.
        # Note that you should name "train_corpus_embedding" and "test_corpus_embedding" for feeding the model.
        self.train(df_train)
        bigramIdx = {}
        for i, item in enumerate(list(self.features)[:feature_num]):
             bigramIdx[item]=i
        train_corpus_embedding=[[0]*feature_num for _ in range(len(df_train['review']))]
        test_corpus_embedding=[[0]*feature_num for _ in range(len(df_test['review']))]
        train_corpus=[['[CLS]']+self.tokenize(document) for document in df_train['review']] 
        test_corpus=[['[CLS]']+self.tokenize(document) for document in df_test['review']] 

        for i, review in enumerate(train_corpus):
             for idx in range(len(review)-1):
                 bigram=(review[idx], review[idx+1])
                 if bigram in bigramIdx:
                     train_corpus_embedding[i][bigramIdx[bigram]]+=1

        for i, document in enumerate(test_corpus):
             for idx in range(len(document)-1):
                 bigram = (document[idx],document[idx+1])
                 if bigram in bigramIdx:
                     test_corpus_embedding[i][bigramIdx[bigram]]+=1
        
        # end your code
        
        # feed converted embeddings to Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(train_corpus_embedding, df_train['sentiment'])
        y_predicted = nb_model.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['sentiment'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    '''
    Here is TA's answer of part 1 for reference only.
    {'a': 0.5, 'saw: 0.25, '.': 0.25}

    Explanation:
    (saw -> a): 2
    (saw -> saw): 1
    (saw -> .): 1
    So the probability of the following word of 'saw' should be 1 normalized by 2+1+1.

    P(I | [CLS]) = 1
    P(saw | I) = 1; count(saw | I) / count(I)
    P(a | saw) = 0.5
    P(saw | a) = 1.0
    P(saw | saw) = 0.25
    P(. | saw) = 0.25
    '''

    # unit test
    test_sentence = {'review': ['I saw a saw saw a saw.']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['saw'])
    print("Perplexity: {}".format(model.compute_perplexity(test_sentence)))
