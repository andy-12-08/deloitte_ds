""" 
I defined the search space for the hyperparameter optimization in this file,
and use the hyperopt library to define the search space.
The search space contains the models and their hyperparameters that I want to optimize.
""" 
from hyperopt import hp
from hyperopt.pyll.base import scope

def create_search_space(num_features):
    """
    Function to dynamically create a search space based on the number of features.
    Args:
        num_features (int): Number of features to consider in SelectKBest.
    Returns:
        dict: The search space dictionary.
    """
    return {
        'model': hp.choice('model', [
            {
                'type': 'RandomForestClassifier',
                'n_estimators': scope.int(hp.quniform('rf_n_estimators', 50, 300, 10)),
                'max_depth': scope.int(hp.quniform('rf_max_depth', 3, 20, 1)),
                'max_features': hp.choice('rf_max_features', ['sqrt', 'log2', None]),
            },
            {
                'type': 'LogisticRegression',
                'C': hp.uniform('lr_C', 0.1, 1.0),
            },
            {
                'type': 'XGBClassifier',
                'n_estimators': scope.int(hp.quniform('xgb_n_estimators', 50, 300, 10)),
                'max_depth': scope.int(hp.quniform('xgb_max_depth', 3, 10, 1)),
                'learning_rate': hp.uniform('xgb_learning_rate', 0.01, 0.3),
            },
            {
                'type': 'LGBMClassifier',
                'n_estimators': scope.int(hp.quniform('lgbm_n_estimators', 50, 300, 10)),
                'max_depth': scope.int(hp.quniform('lgbm_max_depth', -1, 20, 1)),
                'learning_rate': hp.uniform('lgbm_learning_rate', 0.01, 0.3),
                'num_leaves': scope.int(hp.quniform('lgbm_num_leaves', 20, 40, 1)),
            },
            {
                'type': 'CatBoostClassifier',
                'iterations': scope.int(hp.quniform('cat_iterations', 50, 300, 10)),
                'depth': scope.int(hp.quniform('cat_depth', 3, 10, 1)),
                'learning_rate': hp.uniform('cat_learning_rate', 0.01, 0.3),
            },
            {
                'type': 'GaussianNB',
            },
            {
                'type': 'KNeighborsClassifier',
                'n_neighbors': scope.int(hp.quniform('knn_n_neighbors', 3, 15, 1)),
                'weights': hp.choice('knn_weights', ['uniform', 'distance']),
            },
            {
                'type': 'DecisionTreeClassifier',
                'max_depth': scope.int(hp.quniform('dt_max_depth', 3, 20, 1)),
                'min_samples_split': scope.int(hp.quniform('dt_min_samples_split', 2, 10, 1)),
            },
            {
                'type': 'AdaBoostClassifier',
                'n_estimators': scope.int(hp.quniform('ada_n_estimators', 50, 300, 10)),
                'learning_rate': hp.uniform('ada_learning_rate', 0.01, 1.0),
            },
            {
                'type': 'GradientBoostingClassifier',
                'n_estimators': scope.int(hp.quniform('gb_n_estimators', 50, 300, 10)),
                'learning_rate': hp.uniform('gb_learning_rate', 0.01, 0.3),
                'max_depth': scope.int(hp.quniform('gb_max_depth', 3, 10, 1)),
            },
            {
                'type': 'ExtraTreesClassifier',
                'n_estimators': scope.int(hp.quniform('et_n_estimators', 50, 300, 10)),
                'max_depth': scope.int(hp.quniform('et_max_depth', 3, 20, 1)),
                'max_features': hp.choice('et_max_features', ['sqrt', 'log2', None]),
            },
        ]),
        'polynomialfeatures_degree': hp.choice('polynomialfeatures_degree', range(1, 3)),
        'selectkbest_k': hp.choice('selectkbest_k', range(1, num_features + 1)),
    }
