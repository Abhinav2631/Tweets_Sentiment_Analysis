from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def decision_tree_model(X_sample, y_sample):
    dt_model = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=10,
        random_state=42
    )
    dt_model.fit(X_sample, y_sample)
    return dt_model

def random_forest_model(X_sample, y_sample):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_model.fit(X_sample, y_sample)
    return rf_model