import os
import sys
import warnings

warnings.filterwarnings('ignore')

from src.dataset import load_and_engineer_features
from src.models import get_preprocessors, get_models, evaluate_models
from src.utils import generate_visualizations


def main():
    data_path = os.path.join('data', 'heart.csv')
    plots_dir = 'plots'

    print('Veri yükleniyor ve özellik mühendisliği yapılıyor...')
    X, y = load_and_engineer_features(data_path)

    print('Önişlemciler ve modeller hazırlanıyor...')
    prep_A, prep_B = get_preprocessors(X)
    models = get_models()

    print('Modeller değerlendiriliyor (bu işlem bir dakika sürebilir)...')
    results_df = evaluate_models(X, y, prep_A, prep_B, models)

    print('\n--- Model Karşılaştırma Sonuçları ---')
    print(results_df.to_string(index=False))

    results_path = os.path.join(plots_dir, 'evaluation_results.csv')
    results_df.to_csv(results_path, index=False)

    print('\nStacking Ensemble (Senaryo B) için görselleştirmeler üretiliyor...')
    generate_visualizations(X, y, prep_B, models['Stacking Ensemble'], output_dir=plots_dir)

    print('Deney tamamlandı! Sonuçlar ve görseller plots/ klasörüne kaydedildi.')


if __name__ == '__main__':
    main()
