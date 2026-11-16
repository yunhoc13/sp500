"""Model factories and tuning grids for the tree models (shared by 08 and 09)."""

SEED = 42


def make_rf(min_samples_leaf, max_features):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=min_samples_leaf,
                                 max_features=max_features, n_jobs=-1, random_state=SEED)


def make_xgb(max_depth, n_estimators, learning_rate):
    from xgboost import XGBRegressor
    return XGBRegressor(max_depth=max_depth, n_estimators=n_estimators,
                        learning_rate=learning_rate, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=5,
                        reg_lambda=1.0, n_jobs=-1, random_state=SEED, verbosity=0)


MODELS = {
    "rf": (make_rf, {"min_samples_leaf": [3, 5, 10, 20, 40],
                     "max_features": [0.33, 0.6, 1.0]}),
    "xgb": (make_xgb, {"max_depth": [2, 3, 4],
                       "n_estimators": [200, 500],
                       "learning_rate": [0.02, 0.05]}),
}
