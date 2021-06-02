search_params_vect = {
    'ngram_range': [(1, 2), (1, 3), (1, 4)],
    'lowercase': [True],
    'max_df': [0.5],
    'min_df': [2, 5],
    'binary': [True, False],
    }

search_params_tfidf = {'use_idf': (True, False)}

search_params_clf = {
    'bernoulli': {
        'alpha': [0.01, 0.1, 1, 10],
        'fit_prior': [True, False]
        },
    'dtree': {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
        },
    'ridge': {
        'alpha': [0.01, 0.1, 1, 10],
        'normalize': [True, False],
        'fit_intercept': [True, False]
        },
    'etree': {
        'max_depth': [None, 5, 10, 20],
        'n_estimators': [50, 100, 200, 300],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        },
    'knn': {
        'n_neighbors': [5, 10, 20],
        'p': [1, 2],
        'leaf_size': [10, 30, 50],
        },
    'lasso': {
        'alpha': [0.01, 0.1, 1, 10],
        'normalize': [True, False],
        'fit_intercept': [True, False]
        },
    'xgboost': {}
    }