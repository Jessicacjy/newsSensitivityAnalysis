
# coding: utf-8

import numpy as np
import pickle
import logging
np.random.seed(1337)
import re
import codecs
import jieba
import math
import numpy as np
from itertools import product, count  
from heapq import nlargest  
from gensim.models import word2vec  
from sklearn.utils import shuffle
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass


import json
with open("pos.JSON", 'r') as fp:
    pos_text = json.load( fp)


with open("neg.JSON", 'r') as fp:
    neg_text = json.load( fp)


def sentence_split(str_centence):
    list_ret = list()
    for s_str in str_centence.split('。'):
        if '?' in s_str:
            list_ret.extend(s_str.split('？'))
        elif '!' in s_str:
            list_ret.extend(s_str.split('！'))
        else:
            list_ret.append(s_str.strip())
    return list_ret


stopwordset = set()
with open('jieba_dict/stopwords.txt','r',encoding = 'utf-8') as sw:
    for line in sw:
        stopwordset.add(line.strip('\n'))

def seg_art_list(art_list):
    corpus = []
    for title, art in art_list.items():
        corpus.append(" ".join(seg_sent(art)))
    
    return corpus
        
output = open('word_seg.txt','w')
def seg_sent(art):
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'  
    art = art.strip()
    art = re.sub(r, '', art)
    l = []

    seg_list = list(jieba.cut(art, cut_all = False))
    for word in seg_list:
        if word not in stopwordset and word != ' ' and word != "\n" and word != "\n\n":
                output.write(word + ' ')
                l.append(word)
    return l



vectorizer = TfidfVectorizer(sublinear_tf=True,  stop_words='english')
pos_tfidf = seg_art_list(pos_text)
neg_tfidf = seg_art_list(neg_text)


#

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics.scorer import make_scorer
from sklearn import linear_model
from sklearn import metrics
import operator
from time import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts
import timeit
from sklearn.model_selection import GridSearchCV


# In[165]:

#96条正面，123 条负面
X = list(pos_tfidf) + list(neg_tfidf)
y = [1] * 96 + [-1]*123


# ### 模型训练 

# In[305]:

def build_and_evaluate(X, y, classifier=SGDClassifier,outpath = False, grid_Search = True):

    def build(classifier, X, y=None):
        """
        Inner build function that builds a single model.
        """
        if isinstance(classifier, type):
            classifier = classifier()
        if grid_Search:
            parameters = {'vect__max_df': (0.5, 0.75, 1.0),
                        #'vect__max_features': (None, 5000, 10000, 50000),
                        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                        'tfidf__use_idf': (True, False),
                        'tfidf__norm': ('l1', 'l2'),
                        #'clf__alpha': (0.00001, 0.000001),
                        #'clf__penalty': ('l2', 'elasticnet'),
                         #'clf__n_iter': (10, 50, 80),
                }

            pipeline = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', classifier),])
            grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
            print("Performing grid search...")
            print("pipeline:", [name for name, _ in pipeline.steps])
            print("parameters:")
            print(parameters)
            t0 = time()
            grid_search.fit(X, y)
            print("done in %0.3fs" % (time() - t0))
            print()

            print("Best score: %0.3f" % grid_search.best_score_)
            print("Best parameters set:")
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
            #model.fit(X, y)
            return grid_search
        else:
            pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),('tfidf', TfidfTransformer()),('clf', classifier),])
            t0 = time()
            pipeline.fit(X, y)

            print("done in %0.3fs" % (time() - t0))
            return pipeline
    # Begin evaluation

    labels = LabelEncoder()

    y = labels.fit_transform(y)

    print("Building for evaluation")

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)

    model = build(classifier, X_train, y_train)

    y_pred = model.predict(X_test)
    print("classifier result:")
    print(clsr(y_test, y_pred, target_names=['neg', 'pos']))
    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))
    print("confusion matrix")
    print(metrics.confusion_matrix(y_test, y_pred))

    return model

def show_most_informative_features(model, text=None, n=20):
    # Extract the vectorizer and the classifier from the pipeline
    vectorizer = model.named_steps['vect']
    classifier = model.named_steps['clf']
    
    # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
                        "Cannot compute most informative features on {}.".format(
                                                                                 classifier.__class__.__name__
                                                                                 )
                        )
    
    if text is not None:
        # Compute the coefficients for the text
        tvec = model.transform([text]).toarray()
    else:
        # Otherwise simply use the coefficients
        tvec = classifier.coef_
    
    # Zip the feature names with the coefs and sort
    coefs = sorted(
                   zip(tvec[0], vectorizer.get_feature_names()),
                   key=operator.itemgetter(0), reverse=True
                   )
        
                   # Get the top n and bottom n coef, name pairs
                   topn  = zip(coefs[:n], coefs[:-(n+1):-1])
                   
                   # Create the output string to return
                   output = []
                   
                   # If text, add the predicted value to the output.
                   if text is not None:
                       output.append("\"{}\"".format(text))
                       output.append(
                                     "Classified as: {}".format(model.predict([text]))
                                     )
                           output.append("")

# Create two columns with most negative and most positive features.
for (cp, fnp), (cn, fnn) in topn:
    output.append(
                  "{:0.4f}{: >15}    {:0.4f}{: >15}".format(
                                                            cp, fnp, cn, fnn
                                                            )
                  )
                  
    return "\n".join(output)

# ## 1. MultinomialNB

# In[298]:

classifier=MultinomialNB

model_Mb = build_and_evaluate(X,y,classifier,outpath='model')


# ## 2. Logistic Regression

# In[313]:


from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression

model_Lr = build_and_evaluate(X,y,classifier,outpath='model',grid_Search=False)


# ## 3. Stochastic gradient descent

# In[306]:

model_default = build_and_evaluate(X,y,outpath='model',grid_Search=False)


# ## 4. Random Forest

# In[335]:

from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators=20,random_state=5,min_samples_leaf=2)

model_RF = build_and_evaluate(X,y,classifier,outpath='model',grid_Search=False)


# In[353]:

print('负面\t\t','正面')
print(model_RF.predict_proba(neg_tfidf[10:20]))
print(model_RF.predict_proba(pos_tfidf[10:20]))


# In[316]:

model_Lr.predict_proba(neg_tfidf[10:20])


# In[317]:

model_Mb.predict_proba(neg_tfidf[10:20])


# ## 几个模型的训练结果是随机森林效果最佳

# ### 最有决定性的文字：
# 

# #### 这个需要grid_searhch = False

# In[208]:



# In[222]:

words = show_most_informative_features(model)
print(words)


# In[223]:

words_Mb = show_most_informative_features(model_Mb)
print(words_Mb)


# In[284]:

words_Lr = show_most_informative_features(model_Lr)
print(words_Lr)


# In[ ]:



