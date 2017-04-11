from sklearn import model_selection
from sklearn import datasets, metrics, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
#import xgboost as xgb
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#print(train.as_matrix())
pos_train = train[train['is_duplicate'] == 1]
neg_train = train[train['is_duplicate'] == 0]
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
train = pd.concat([pos_train, neg_train])

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        q1words[word] = 1
    for word in str(row['question2']).lower().split():
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, df):
        df['z_len1'] = df.question1.map(lambda x: len(str(x)))
        df['z_len2'] = df.question2.map(lambda x: len(str(x)))
        df['z_len_diff'] = df['z_len1'] - df['z_len2']
        df['z_word_len1'] = df.question1.map(lambda x: len(str(x).split()))
        df['z_word_len2'] = df.question2.map(lambda x: len(str(x).split()))
        df['z_word_len_diff'] =  df['z_word_len1'] -  df['z_word_len2']
        df['z_word_match'] = df.apply(word_match_share, axis=1, raw=True)
        df['zavg_world_len1'] = df['z_len1'] / df['z_word_len1']
        df['zavg_world_len2'] = df['z_len2'] / df['z_word_len2']
        df['zavg_world_len_diff'] = df['zavg_world_len1'] - df['zavg_world_len2']
        df['zexactly_same'] = (df['question1'] == df['question2']).astype(int)
        df['zduplicated'] = df.duplicated(['question1','question2']).astype(int)
        df = df.fillna(0.0)
        col = [c for c in df.columns if c[:1]=='z']
        return df[col]

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)

tfidf = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 1), max_features=1000)
reg = Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst', Pipeline([('cst1', cust_regression_vals())])),
                        ('txt1', Pipeline([('s1', cust_txt_col(key='question1')), ('tfidf1', tfidf)])),
                        ('txt2', Pipeline([('s2', cust_txt_col(key='question2')), ('tfidf2', tfidf)])),
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        'txt1': 1.0,
                        'txt2': 1.0
                        },
                n_jobs = -1
                ))])

df = reg.fit_transform(train[['question1','question2']])
x_train, x_valid, y_train, y_valid = train_test_split(df, train['is_duplicate'], test_size=0.2, random_state=0)
lr = GradientBoostingRegressor(n_estimators=5000, random_state=1)
lr.fit(x_train,y_train)
#lr.predict(x_valid.toarray())
score = metrics.mean_squared_error(lr.predict(x_valid.toarray()), y_valid)
print(score)
'''
df = reg.transform(test[['question1','question2']])
sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub['is_duplicate'] = lr.predict(df.toarray())
sub.to_csv('latest_trial.csv', index=False)'''

'''
df = []
params = {}
params["objective"] = "binary:logistic"
params['eval_metric'] = 'logloss'
params["eta"] = 0.2
params["subsample"] = 0.7
params["min_child_weight"] = 1
params["colsample_bytree"] = 0.7
params["max_depth"] = 5
params["silent"] = 1
params["seed"] = 12357

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
bst = xgb.train(params, d_train, 100, watchlist, early_stopping_rounds=10, verbose_eval=10)
x_train, x_valid, y_train, y_valid, d_train, d_valid = [None, None, None, None, None, None]
df = reg.transform(test[['question1','question2']])
sub = pd.DataFrame()
sub['test_id'] = test['test_id']
sub['is_duplicate'] = bst.predict(xgb.DMatrix(df))

sub.to_csv('z09_submission_xgb_01.csv', index=False)'''