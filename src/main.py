'''
Define search parameters and run them here
'''

# %%
import gc
import pickle

import numpy as np
import pandas as pd
import ndjson

# infrastructure
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from scipy.sparse import hstack

# preprocessing
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# in house scripts
from gridsearching import GridSearchClassifier
from parameter_grid import search_params_clf

# %%
# load data
edh = pd.read_csv('/home/jan/ancient-classifier/data/210416_certain_Y/edh_56110.csv')


# %% 
# clearing NaN
edh_ = edh.dropna(subset=['clean_text_conservative'])
edh_.isna().sum()


# %%
# minimal run

# edh_min = edh_.iloc[0:100, :]

vectorizer = CountVectorizer()
transformer = TfidfTransformer()

counts_min = vectorizer.fit_transform(edh_['clean_text_conservative'])
tfidf_min = transformer.fit_transform(counts_min)

# other features
features_of_interest = ['province_label_clean', 'country_clean']

ct = ColumnTransformer([
    (
        'scale',
        StandardScaler(),
        make_column_selector(dtype_include=np.number)
    ),
    (
        'onehot',
        OneHotEncoder(),
        make_column_selector(dtype_include=object)
    )
])

features_min = ct.fit_transform(edh_[features_of_interest])

# %%
# stack arrays
# X_min = np.hstack(
#     (
#         tfidf_min.toarray(),
#         features_min.toarray()
#     )
# )

# %%
# stack sparse matrices
X_vect = counts_min

X_min = tfidf_min


X_max = hstack(
    (
        tfidf_min,
        features_min
    )
)

# %%
# get weights
province_certainity_weights = np.array(
    [1.0 if label == 'Certain' else 0.5  for label in edh_.province_label_certainty]
)

# %%
# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_max, edh_.type_of_inscription_clean)

# %%
# run funk
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score

scoring = {
    'accuracy': make_scorer(accuracy_score),
    # 'precision': make_scorer(precision_score, average='weighted'),
    # 'recall': make_scorer(recall_score, average='weighted')
}

gs = GridSearchClassifier(
    clfs=['ridge'],
    resamplers=None,
    scoring=scoring,
    n_jobs=8,
    clf_param_grid=search_params_clf
)

res = {}
res['ridge_over'] = gs.grid_search_1_clf(
    X_train, y_train,
    clf_tag='ridge', sampler_tag='over',
    kwargs_cv={'error_score': 'raise'}
    )

res['ridge_under'] = gs.grid_search_1_clf(
    X_train, y_train,
    clf_tag='ridge', sampler_tag='under',
    kwargs_cv={'error_score': 'raise'}
    )

res['ridge_smote'] = gs.grid_search_1_clf(
    X_train, y_train,
    clf_tag='ridge', sampler_tag='smote',
    kwargs_cv={'error_score': 'raise'}
    )

# res['lasso'] = gs.grid_search_1_clf(
#     X_train, y_train,
#     clf_tag='lasso', sampler_tag=None,
#     kwargs_cv={'error_score': 'raise'}
#     )

# res['xgboost'] = gs.grid_search_1_clf(
#     X_train, y_train,
#     clf_tag='xgboost', sampler_tag=None,
#     kwargs_cv={'error_score': 'raise'}
#     )

with open('/home/jan/ancient-classifier/res/210505_res_ridge_max_resample.pickle', 'wb') as fout:
    pickle.dump(res, fout)

print(res)

gc.collect()

# %%
