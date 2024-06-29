import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate


# Load the Iris dataset and set the features and labels
iris = load_iris()
features, labels = iris.data, iris.target

# Preprocessing using standard scalar
standard_scalar = StandardScaler()
features = standard_scalar.fit_transform(features)

# Create the models & tune the hyperparameters
naive_bayes = GaussianNB()
svm = SVC(probability=True)
# Tuning the hyperparameters for Random Forest & XGBoost to prevent overfitting
random_forest = RandomForestClassifier(n_estimators=10, max_depth=5, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=True)
Xgboost = xgb.XGBClassifier(n_estimators=5, use_label_encoder=False, eval_metric='logloss', max_depth=3, learning_rate=0.1, reg_alpha=0.1, reg_lambda=0.1)
knn = KNeighborsClassifier()

# Create the five fold cross validation
cross_validation = KFold(n_splits=5, shuffle=True, random_state=53)

# Run cross-validation and collect the evaluation metrics for each model
total_evaluation_metrics = {}


# Create the evaluation metrics: accuracy, f1-score, roc auc
evaluation_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score, average='macro', labels=np.unique(labels)),
    'roc_auc': make_scorer(roc_auc_score, average='macro', needs_proba=True, multi_class='ovr')
}


# Gaussian Naive Bayes cross validation & evaluation metrics
cross_val_model = cross_validate(naive_bayes, features, labels, cv=cross_validation, scoring=evaluation_metrics, return_train_score=True)
total_evaluation_metrics["Naive Bayes"] = {
    'Training: Accuracy': np.mean(cross_val_model['train_accuracy']),
    'Training F1 Score': np.mean(cross_val_model['train_f1_score']),
    'Training ROC AUC': np.mean(cross_val_model['train_roc_auc']),
    'Test Accuracy': np.mean(cross_val_model['test_accuracy']),
    'Test F1 Score': np.mean(cross_val_model['test_f1_score']),
    'Test ROC AUC': np.mean(cross_val_model['test_roc_auc'])
}

# Support Vector Machine cross validation & evaluation metrics
cross_val_model = cross_validate(svm, features, labels, cv=cross_validation, scoring=evaluation_metrics, return_train_score=True)
total_evaluation_metrics["Support Vector Machine"] = {
    'Training: Accuracy': np.mean(cross_val_model['train_accuracy']),
    'Training F1 Score': np.mean(cross_val_model['train_f1_score']),
    'Training ROC AUC': np.mean(cross_val_model['train_roc_auc']),
    'Test Accuracy': np.mean(cross_val_model['test_accuracy']),
    'Test F1 Score': np.mean(cross_val_model['test_f1_score']),
    'Test ROC AUC': np.mean(cross_val_model['test_roc_auc'])
}

# Random Forest cross validation & evaluation metrics
cross_val_model = cross_validate(random_forest, features, labels, cv=cross_validation, scoring=evaluation_metrics, return_train_score=True)
total_evaluation_metrics["Random Forest"] = {
    'Training: Accuracy': np.mean(cross_val_model['train_accuracy']),
    'Training F1 Score': np.mean(cross_val_model['train_f1_score']),
    'Training ROC AUC': np.mean(cross_val_model['train_roc_auc']),
    'Test Accuracy': np.mean(cross_val_model['test_accuracy']),
    'Test F1 Score': np.mean(cross_val_model['test_f1_score']),
    'Test ROC AUC': np.mean(cross_val_model['test_roc_auc'])
}

# XGBoost cross validation & evaluation metrics
cross_val_model = cross_validate(Xgboost, features, labels, cv=cross_validation, scoring=evaluation_metrics, return_train_score=True)
total_evaluation_metrics["XGBoost"] = {
    'Training: Accuracy': np.mean(cross_val_model['train_accuracy']),
    'Training F1 Score': np.mean(cross_val_model['train_f1_score']),
    'Training ROC AUC': np.mean(cross_val_model['train_roc_auc']),
    'Test Accuracy': np.mean(cross_val_model['test_accuracy']),
    'Test F1 Score': np.mean(cross_val_model['test_f1_score']),
    'Test ROC AUC': np.mean(cross_val_model['test_roc_auc'])
}

# K-Nearest Neighbors cross validation & evaluation metrics
cross_val_model = cross_validate(knn, features, labels, cv=cross_validation, scoring=evaluation_metrics, return_train_score=True)
total_evaluation_metrics["K-Nearest Neighbors"] = {
    'Training: Accuracy': np.mean(cross_val_model['train_accuracy']),
    'Training F1 Score': np.mean(cross_val_model['train_f1_score']),
    'Training ROC AUC': np.mean(cross_val_model['train_roc_auc']),
    'Test Accuracy': np.mean(cross_val_model['test_accuracy']),
    'Test F1 Score': np.mean(cross_val_model['test_f1_score']),
    'Test ROC AUC': np.mean(cross_val_model['test_roc_auc'])
}



# Store evaluation metrics and print it out as a table
evaluation_metrics_table = []
for model, val in total_evaluation_metrics.items():
    row = [model]
    for score_name, score_value in val.items():
        row.append(f"{score_value:.4f}")
    evaluation_metrics_table.append(row)

# Print the table
print(tabulate(evaluation_metrics_table, headers=["Model", "Training: Accuracy", "Training F1 Score", "Training ROC AUC", "Test Accuracy", "Test F1 Score", "Test ROC AUC"], tablefmt="grid"))
