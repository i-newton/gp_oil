from utils import get_fold
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVR
import xgboost as xgb
MAX_TOWERS = 6
from sklearn.neural_network import MLPRegressor


class Ensembler:

    def __init__(self):
        self.pred_names = ["lasso", "rtree", "xgboost"]
        ridge = Ridge(random_state=17)
        rtree = RandomForestRegressor(n_jobs=-1,
                                      random_state=17)
        svr = LinearSVR(random_state=17)
        xgboost = xgb.XGBRegressor(random_state=17, n_jobs=-1)
        nn = MLPRegressor(max_iter=500, random_state=17)
        et = ExtraTreesRegressor(n_jobs=-1, n_estimators=100, random_state=17)
        lasso = Lasso(random_state=17,tol=0.001)

        self.predictors = [lasso, rtree, xgboost]
        self.stack_predictor = xgb.XGBRegressor(random_state=17, n_jobs=-1)

    def _get_train_preds(self, X, y):
        predicts = []
        for cl in self.predictors:
            predicts.append(
                cross_val_predict(cl, X, y, n_jobs=-1, cv=get_fold()))
        return pd.DataFrame(np.vstack(predicts).transpose(), index=y.index,
                            columns=self.pred_names)

    def fit(self, X_train, y_train):
        #first get predictions
        predicts = self._get_train_preds(X_train, y_train)
        for rgr in self.predictors:
            rgr.fit(X_train, y_train)
        #fit stack regressor
        self.stack_predictor.fit(predicts, y_train)

    def predict(self, X_test):
        test_predicts = []
        for rgr in self.predictors:
            pr = rgr.predict(X_test)
            test_predicts.append(pr)
            # add svd predicts
        test_predictions = pd.DataFrame(np.vstack(test_predicts).transpose(),
                                        index=X_test.index,
                                        columns=self.pred_names)
        return self.stack_predictor.predict(test_predictions)


