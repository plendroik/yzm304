import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import pandas as pd


def get_preprocessors(X):
    """İki farklı önişleme senaryosu (A: Baseline, B: Gelişmiş) oluşturur."""
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Senaryo A (Baseline): Basit ortalama doldurma + StandardScaler
    numeric_transformer_A = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer_A = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor_A = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_A, numeric_features),
            ('cat', categorical_transformer_A, categorical_features)
        ])

    # Senaryo B (Gelişmiş): MICE (IterativeImputer) + RobustScaler
    numeric_transformer_B = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=42, max_iter=10)),
        ('scaler', RobustScaler())
    ])

    categorical_transformer_B = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor_B = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_B, numeric_features),
            ('cat', categorical_transformer_B, categorical_features)
        ])

    return preprocessor_A, preprocessor_B


def get_models():
    """Temel modelleri ve Stacking Ensemble modelini tanımlar."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'MLP': MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(64, 32))
    }

    # Stacking Ensemble: Seviye 0 (RF, XGBoost, SVM) → Seviye 1 (Lojistik Regresyon)
    level0 = [
        ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
        ('svm', SVC(probability=True, random_state=42))
    ]
    level1 = LogisticRegression(random_state=42)
    models['Stacking Ensemble'] = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

    return models


def evaluate_models(X, y, preprocessor_A, preprocessor_B, models):
    """Tüm modelleri her iki senaryo ile 5-Fold Stratified CV kullanarak değerlendirir."""
    results = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        for scenario_name, preprocessor in [('Senaryo A', preprocessor_A), ('Senaryo B', preprocessor_B)]:
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

            scoring = {'accuracy': 'accuracy', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}
            cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

            results.append({
                'Model': name,
                'Senaryo': scenario_name,
                'Accuracy': np.mean(cv_results['test_accuracy']),
                'Recall': np.mean(cv_results['test_recall']),
                'F1-Score': np.mean(cv_results['test_f1']),
                'AUC-ROC': np.mean(cv_results['test_roc_auc'])
            })

    return pd.DataFrame(results)
