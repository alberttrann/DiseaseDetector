# Best hyperparameters found through tuning

BEST_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 2,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    },
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'num_leaves': 15,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    },
    'catboost': {
        'iterations': 200,
        'depth': 8,
        'learning_rate': 0.01,
        'random_state': 42,
        'verbose': 0,
    },
}
