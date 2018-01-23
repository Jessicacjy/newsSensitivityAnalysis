
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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split as tts
import sklearn
import timeit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass


# ## 打开正面和负面的新闻
# 格式是 {标题：新闻内容}


import json
with open("pos.JSON", 'r') as fp:
    pos_text = json.load( fp)
with open("neg.JSON", 'r') as fp:
    neg_text = json.load( fp)


#建立停词
stopwordset = set()
with open('sa/jieba_dict/stopwords.txt','r',encoding = 'utf-8') as sw:
    for line in sw:
        stopwordset.add(line.strip('\n'))

#分词wrapper
def seg_art_list(art_list):
    corpus = []
    for title, art in art_list.items():
        corpus.append(" ".join(seg_sent(art)))
    return corpus

#分词，去标点符号，去停词，去英文，去数字
#art： 单篇文章
#return：所有的单词
output = open('word_seg.txt','w')
def seg_sent(art):
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'  
    art = art.strip()
    art = re.sub(r, '', art)
    art = re.sub("[a-z]","",art)
    art = re.sub("[0-9]","",art)
    l = []

    seg_list = list(jieba.cut(art, cut_all = False))
    for word in seg_list:
        if word not in stopwordset and word != ' ' and word != "\n" and word != "\n\n":
                output.write(word + ' ')
                l.append(word)
    return l

#建立和保存正面新闻的 分词
#建立和保存负面新闻的 分词
pos_tfidf = seg_art_list(pos_text)
neg_tfidf = seg_art_list(neg_text)
with open("pos_tfidf.JSON", 'w') as fp:
    json.dump(pos_tfidf, fp)
with open("neg_tfidf.JSON", 'w') as fp:
    json.dump(neg_tfidf, fp)

#Y： 1为正面，-1为负面
X = list(pos_tfidf) + list(neg_tfidf)
y = [1] * len(pos_tfidf) + [-1]*len(neg_tfidf)

# ### 模型训练 

def build_and_evaluate(X, y, classifier=SGDClassifier,outpath = False, grid_Search = True, prameters = False):

    def build(classifier, X, y=None):
        """
        Inner build function that builds a single model.
        """
        if isinstance(classifier, type):
            classifier = classifier()
        if grid_Search:
            if not parameters:
                parameters = {'vect__max_df': (0.5, 0.75, 1.0),
                            'vect__max_features': (None, 5000, 10000, 50000),
                            'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                            'tfidf__use_idf': (True, False),
                            'tfidf__norm': ('l1', 'l2'),
                            'clf__alpha': (0.00001, 0.000001),
                            'clf__penalty': ('l2', 'elasticnet'),
                            'clf__n_iter': (10, 50, 80),
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
            if outpath:
                with open(outpath, 'wb') as f:
                    pickle.dump(grid_search, f, protocol = 2)
            print("Model written out to {}".format(outpath))
            return grid_search

        else:
            select = sklearn.feature_selection.SelectKBest(k=100)
            
            pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), min_df=20,max_df = 0.9)),
                                 ('tfidf', TfidfTransformer()),
                                 ('feature_selection', select),
                                 ('clf', classifier)])
            cv = CountVectorizer(ngram_range=(1,2), min_df=20,max_df = 0.9)
            X = cv.fit_transform(X)
            tfidf = TfidfVectorizer()
            X = tfidf.transform(X)
            feature_names = tfidf.get_feature_names()
            
            t0 = time()
            model = classifier.fit(X,y)
            

            print("done in %0.3fs" % (time() - t0))
            
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
            pickle.dump(model, f, protocol = 2)

        print("Model written out to {}".format(outpath))
    print("confusion matrix")
    print(metrics.confusion_matrix(y_test, y_pred))

    return model


# ## 4. Random Forest

if __name__ == "__main__":



classifier= RandomForestClassifier(n_estimators=20,random_state=5,min_samples_leaf=2)

model_RF = build_and_evaluate(X,y,classifier,outpath='model',grid_Search=False)


a = model_RF.steps[1]




for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    r = rf.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test))
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))
        scores[names[i]].append((acc-shuff_acc)/acc)
print "Features sorted by their score:"
print sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores.items()], reverse=True)



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

# In[23]:


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













