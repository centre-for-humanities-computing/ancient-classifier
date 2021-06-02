"""
A script for conducting grid search using sklearn
TODO change title
"""
# %%
# pipeline
from imblearn.pipeline import Pipeline

# classifiers
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier, Lasso
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# resampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

# utility gridsearch
from sklearn.model_selection import GridSearchCV


# %%
class GridSearchClassifier():

    def __init__(self,
                clfs=['bernoulli', 'dtree', 'ridge', 'etree', 'knn', 'lasso', 'xgboost'],
                resamplers=['over', 'under', 'smote'],
                cv=5,
                scoring='accuracy',
                n_jobs=4,
                clf_param_grid=None, 
                resampling_param_grid=None,
                ) -> None:

        if isinstance(clfs, str):
            clfs = [clfs]
        if isinstance(resamplers, str):
            resamplers = [resamplers]

        self.clfs = clfs
        self.resamplers = resamplers
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.clf_param_grid = clf_param_grid
        self.resampling_param_grid = resampling_param_grid


    @staticmethod
    def get_clf(clf_tag):
        '''Instantiate classifier based on tag

        Parameters
        ----------
        clf_tag : str
            Tag of classifer to instantiate


        Returns
        -------
        sklearn classifier
            Instantiated classifier class

        Raises
        ------
        ValueError
            In case desired `clf_tag` is not found
        '''
        # instantiate classifiers
        clf_dict = {"bernoulli": BernoulliNB,
                    "dtree": DecisionTreeClassifier,
                    "ridge": RidgeClassifier,
                    'etree': ExtraTreesClassifier,
                    'knn': KNeighborsClassifier,
                    'lasso': Lasso,
                    'xgboost': XGBClassifier
                    }

        # delive ordered classifier
        if clf_tag in clf_dict:
            clf = clf_dict[clf_tag]
        else:
            raise ValueError(f"{clf_tag} is not a valid classifier")

        return clf

    @staticmethod
    def get_resampler(sampler_tag):
        '''Instentiate sampler based on tag

        Parameters
        ----------
        sampler_tag : str
            Tag of sampler to instantiate

        Returns
        -------
        imblearn resampler
            Instantiated resampling class

        Raises
        ------
        ValueError
            In case desired `sampler` is not found
        '''

        resampling_dict = {
            "over": RandomOverSampler,
            "under": RandomUnderSampler,
            "smote": SMOTE
            }
        
        if sampler_tag in resampling_dict:
            sampler = resampling_dict[sampler_tag]
        else:
            raise ValueError(f"{sampler_tag} is not a valid resampler")

        return sampler


    def grid_search_1_clf(self, X, y, clf_tag, sampler_tag,
    kwargs_sampler={}, kwargs_clf={}, kwargs_cv={}, kwargs_fit={}):
        '''
        Grid search with 1 classifier & it's parameter grid,
        using one resampling strategy.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values

        clf_tag : str
            Tag to instantiate classifier.
            Together with it's parameter grid, if specified.

        sampler_tag : str
            Tag to instantiate resampler.
            Single resampling strategy can be used.

        **kwargs
            Any futher parameters 
            passed to either classifier, or resampler
        '''

        # instantiate classifier
        clf = self.get_clf(clf_tag)

        if self.resamplers:
            # opt1: resample training data 
            #       using ONE sampling strategy
            sampler = self.get_resampler(sampler_tag)
            pipe = Pipeline([
                ('sampling', sampler(**kwargs_sampler)),
                ('clf', clf(**kwargs_clf))
                ])

        else:
            # opt2: use training data as is
            pipe = Pipeline([
                ('clf', clf(**kwargs_clf))
                ])

        parameters = {}
        if self.clf_param_grid:
            # compile parameter grid
            for k in self.clf_param_grid[clf_tag]:
                # extract parameter grid of ONE `clf_tag`
                parameters['clf__'+k] = self.clf_param_grid[clf_tag][k]

        # to prevent mistakes when passing multiple scorers
        # gridsearch will optimize accuracy and just monitor the rest
        if not isinstance(self.scoring, str):
            kwargs_cv.update({'refit': 'accuracy'}) 

        gs_clf = GridSearchCV(
            pipe,
            parameters,
            scoring=self.scoring,
            cv=self.cv,
            verbose=True,
            n_jobs=self.n_jobs,
            **kwargs_cv
            )

        print(f'[info] fitting {clf_tag}, with resampler {sampler_tag}')
        clf_res = gs_clf.fit(X, y, **kwargs_fit)

        return clf_res.cv_results_
        # return (clf_res.best_score_, clf_res.best_params_, clf_res)


    @staticmethod
    def print_report(results):
        # TODO make it work when no resampling grid search is ordered
        print("\n\nThe grid search is completed the results were:")
        for rs in results:
            print(f"\nUsing the resampling method: {rs}")
            for c in results[rs]:
                score, best_params, t = results[rs][c]
                print(f"\tThe best fit of the clf: {c}, " +
                    f"obtained a score of {round(score, 4)}, \
                        with the parameters:")
                for p in best_params:
                    print(f"\t\t{p} = {best_params[p]}")


    def fit(self, X, y, **kwargs):

        results = {}
        
        if self.resamplers:
            # iterate sampling techniques
            for sampler_tag in self.resamplers:
                best_model_per_clf = {}
                # iterate classifiers
                for clf_tag in self.clfs:
                    # grid search with one classifier
                    one_clf_res = self.grid_search_1_clf(
                        X, y,
                        clf_tag, sampler_tag,
                        **kwargs
                        )
                    
                    # save best model per classifier
                    best_model_per_clf[clf_tag] = one_clf_res

                # save best model per sampling strategy 
                results[sampler_tag] = best_model_per_clf
        
        else:
            for clf_tag in self.clfs:
                # grid search with one classifier
                one_clf_res = self.grid_search_1_clf(
                    X, y,
                    clf_tag, sampler_tag=None,
                    **kwargs
                    )
                
                # save best model per classifier
                results[clf_tag] = one_clf_res

        return results
