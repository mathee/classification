MODEL: SUPPORT_VECTOR_MACHINE
BEST PARAMS: {'C': 0.3, 'gamma': 10, 'kernel': 'rbf'}
SCORE: 0.5064596273291926

GRIDSEARCH: GridSearchCV(cv=5, error_score='raise',
       estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'C': [0.3, 0.31], 'kernel': ['rbf'], 'gamma': [10, 11]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring='accuracy', verbose=0)

        