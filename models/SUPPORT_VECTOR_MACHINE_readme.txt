MODEL: SUPPORT_VECTOR_MACHINE
BEST PARAMS: {'C': 0.11, 'gamma': 9, 'kernel': 'rbf'}
SCORE: 0.56625

GRIDSEARCH: GridSearchCV(cv=5, error_score='raise-deprecating',
       estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False),
       fit_params=None, iid='warn', n_jobs=None,
       param_grid={'C': [0.08, 0.09, 0.1, 0.11], 'kernel': ['rbf'], 'gamma': [6, 7, 8, 9, 10, 11]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring='accuracy', verbose=0)

        