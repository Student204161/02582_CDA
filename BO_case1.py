# main.py
import wandb
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import yaml


def objective_elasticnet():
    with wandb.init(config=wandb.config,mode="offline"):
        config = wandb.config
        alpha = config.alpha
        l1_ratio = config.l1_ratio

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mean_score = -np.sqrt(-np.mean(scores))
        wandb.log({"neg_root_mean_mse": mean_score})
        print(mean_score)
        return mean_score


# Load data and preprocess
data = pd.read_csv('Case1Data.csv')
data = data.drop('C_02', axis=1)
y = np.array(data['y'])
X_nan = np.array(data.drop('y', axis=1))

from sklearn.linear_model import LinearRegression

def fill_nan(X, method='mean'):
    """Fill NaN values in a 2D array using the specified method"""
    if method == 'mean':
        column_means = np.nanmean(X, axis=0)
        nan_matrix = np.isnan(X)
        X[nan_matrix] = np.take(column_means, np.where(nan_matrix)[1])
    elif method == 'regression':
        nan_matrix = np.isnan(X)
        column_means = np.nanmean(X, axis=0)
        X[nan_matrix] = np.take(column_means, np.where(nan_matrix)[1])

        for col in range(X.shape[1]):
            missing_rows = np.where(nan_matrix[:, col])[0]

            if len(missing_rows) > 0:
                corr_matrix = np.corrcoef(X.T)
                corr_matrix = np.nan_to_num(corr_matrix)

                correlation = np.abs(corr_matrix[col])
                sorted_indices = np.argsort(-correlation)
                correlated_indices = [idx for idx in sorted_indices if idx != col][:5]

                valid_rows = ~np.any(nan_matrix[:, correlated_indices], axis=1)
                X_train_local, y_train_local = X[valid_rows][:, correlated_indices], X[valid_rows][:, col]

                if len(X_train_local) > 0:
                    model = LinearRegression()
                    model.fit(X_train_local, y_train_local)

                    X_test = X[missing_rows][:, correlated_indices]
                    X[missing_rows, col] = model.predict(X_test)

    return X

X = fill_nan(X_nan, method='mean')
X_train_global, X_test, y_train_global, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train_global
y_train = y_train_global

if __name__ == "__main__":
    # # Initialize W&B project
    wandb.init(mode="offline")

    #load sweep config
    with open('sweep.yaml', 'r') as file:
        sweep_config = yaml.safe_load(file)
    
    # define sweep
    sweep_id = wandb.sweep(sweep_config, project="case1")

    wandb.agent(sweep_id, function=objective_elasticnet, count=2)