import pandas as pd
import numpy as np


def load_and_engineer_features(file_path):
    """Veri setini yükler ve yeni özellikler türetir."""
    df = pd.read_csv(file_path)

    # Özellik 1: Klinik Oran (Kolesterol / Maksimum Kalp Hızı)
    df['ClinicalRatio'] = df['Cholesterol'] / (df['MaxHR'] + 1e-6)

    # Özellik 2: Şiddet Skoru
    # Göğüs ağrısı tiplerini risk sırasına göre numaralandır (ASY en yüksek riskli)
    cp_mapping = {'NAP': 1, 'ATA': 2, 'TA': 3, 'ASY': 4}
    # Egzersiz anjinini 1/0 olarak kodla
    ea_mapping = {'N': 0, 'Y': 1}

    df['ChestPainMapped'] = df['ChestPainType'].map(cp_mapping).fillna(0)
    df['ExerciseAnginaMapped'] = df['ExerciseAngina'].map(ea_mapping).fillna(0)

    df['SeverityScore'] = df['ChestPainMapped'] + df['ExerciseAnginaMapped']

    # Ara sütunları sil, orijinal kategorik sütunlar one-hot encoding için kalsın
    df.drop(columns=['ChestPainMapped', 'ExerciseAnginaMapped'], inplace=True)

    # Kolesterol değeri 0 olan örnekler aslında eksik veridir
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)

    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    return X, y
