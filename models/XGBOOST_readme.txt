MODEL: XGBOOST
BEST PARAMS: {}
GRID_SCORE: 0.8035914702581369
BEST_ESTIMATOR_ACCURACY: 0.8888888888888888
BEST_ESTIMATOR_PRECISION: 0.8957654723127035
BEST_ESTIMATOR_RECALL: 0.804093567251462
CONFUSION_MATRIX: 
[[517  32]
 [ 67 275]]

GRIDSEARCH: GridSearchCV(cv=5, error_score='raise-deprecating',
       estimator=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1),
       fit_params=None, iid='warn', n_jobs=None, param_grid={},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring='accuracy', verbose=0)

        