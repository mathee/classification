MODEL: RANDOM_FOREST
BEST PARAMS: {'max_depth': 12, 'n_estimators': 17}
GRID_SCORE: 0.8226711560044894
BEST_ESTIMATOR_ACCURACY: 0.9562289562289562
BEST_ESTIMATOR_PRECISION: 0.9690402476780186
BEST_ESTIMATOR_RECALL: 0.9152046783625731
CONFUSION_MATRIX: 
[[539  10]
 [ 29 313]]

GRIDSEARCH: GridSearchCV(cv=5, error_score='raise-deprecating',
       estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
            oob_score=False, random_state=42, verbose=0, warm_start=False),
       fit_params=None, iid='warn', n_jobs=None,
       param_grid={'n_estimators': [16, 17], 'max_depth': [11, 12]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring='accuracy', verbose=0)

        