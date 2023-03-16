
#%% Imports
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression

# import xgboost
from xgboost import XGBClassifier
#%% Data Loading
data_dir = Path(__file__).parent.parent / 'data'
df = pd.read_csv(data_dir / 'train.csv')
df_test = pd.read_csv(data_dir / 'test.csv')

#%% Modelling
features = [
    'SeedA',
    'SeedB',
    'WinRatioA',
    'GapAvgA',
    'WinRatioB',
    'GapAvgB',
    'SeedDiff',
    'WinRatioDiff',
    'GapAvgDiff'
]

def rescale(features, df_train, df_val, df_test=None):

    zscore = StandardScaler()
    df_train[features] = zscore.fit_transform(df_train[features])
    df_val[features] = zscore.transform(df_val[features])

    if df_test is not None:
        df_test[features] = zscore.transform(df_test[features])
        
    return df_train, df_val, df_test

# Cross Validation
# Validate on season n, for n in the 10 last seasons.
# Train on earlier seasons
# Pipeline support classification (predict the team that wins) and regression (predict the score gap)

def run_prediction(df, df_test, plot=False, verbose=0, mode="reg"):
    seasons = df['Season'].unique()
    cvs = []
    pred_tests = []
    target = "ScoreDiff" if mode == "reg" else "WinA"

    for season in seasons[10:]:
        if verbose:
            print(f'\nValidating on season {season}')

        # split into previous seasons (train), season n (val), and test (current season)
        df_train = df[df['Season'] < season].reset_index(drop=True).copy()
        df_val = df[df['Season'] == season].reset_index(drop=True).copy()
        df_test = df_test.copy()

        # z-score
        df_train, df_val, df_test = rescale(features, df_train, df_val, df_test)

        # model 
        model = XGBClassifier(n_estimators=1000, max_depth=3)
        # parameter tuning
        params = {
            'n_estimators': [1000, 2000],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        grid = GridSearchCV(model, params, scoring='neg_log_loss', cv=5, verbose=1)
        grid.fit(df_train[features], df_train[target])

        model = grid.best_estimator_
        model.fit(df_train[features], df_train[target], eval_set=[(df_val[features], df_val[target])], verbose=verbose)
        pred = model.predict_proba(df_val[features])[:, 1]
        pred_test = model.predict_proba(df_test[features])[:, 1]
        pred_tests.append(pred_test)

        if plot:
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(pred, df_val['ScoreDiff'].values, s=5)
            plt.grid(True)
            plt.subplot(1, 2, 2)
            sns.histplot(pred)
            plt.show()
        
        loss = log_loss(df_val['WinA'].values, pred)
        cvs.append(loss)
        if verbose:
            print(f'\t -> Scored {loss:.3f}')
        
    print(f'\n Local CV is {np.mean(cvs):.3f}')
    
    return pred_tests

pred_tests = run_prediction(df, df_test, plot=False, verbose=1, mode="cls")

# Submission
pred_test = np.mean(pred_tests, 0)
sub = df_test[['ID', 'Pred']].copy()
sub['Pred'] = pred_test
sub.to_csv(data_dir / 'submission.csv', index=False)
