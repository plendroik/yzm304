import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve


def generate_visualizations(X, y, preprocessor, stacking_model, output_dir='plots'):
    """Çapraz doğrulama (OOF) tahminleri kullanarak görselleştirmeler üretir.
    Veri sızıntısını (data leakage) önlemek için train=test yaklaşımı kullanılmaz."""
    os.makedirs(output_dir, exist_ok=True)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', stacking_model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Her örnek, test katmanındayken tahmin edilir (out-of-fold)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, method='predict')
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')[:, 1]

    # 1. Karışıklık Matrisi (Confusion Matrix)
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Karışıklık Matrisi — Stacking Ensemble (5-Fold CV)')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # 2. ROC Eğrisi
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc_val = roc_auc_score(y, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Yanlış Pozitif Oranı (FPR)')
    plt.ylabel('Doğru Pozitif Oranı (TPR)')
    plt.title('ROC Eğrisi — Stacking Ensemble (5-Fold CV)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
    plt.close()

    # 3. Özellik Önemi (Meta-Model Ağırlıkları)
    # Katsayı çıkarımı için tüm veri üzerinde eğit
    pipeline.fit(X, y)
    meta_model = pipeline.named_steps['classifier'].final_estimator_
    coefs = meta_model.coef_[0]
    base_models = [name for name, _ in stacking_model.estimators]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=coefs, y=base_models, palette='viridis')
    plt.title('Özellik Önemi (Meta-Model Ağırlıkları)')
    plt.xlabel('Ağırlık (Katsayı)')
    plt.ylabel('Temel Modeller')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
    plt.close()
