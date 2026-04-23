import torch
import numpy as np
import os
from torchvision import models
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def extract_features(model, loader, device, limit=None):
    model.eval()
    features = []
    labels_list = []
    count = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            # VGG16 feature extractor
            feat = model.features(inputs)
            # (1, 1) pooling ile ozellik boyutunu 25088'den 512'ye dusuruyoruz.
            # Bu, SVM egitimini cok daha hizli hale getirir.
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            feat = torch.flatten(feat, 1)
            features.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())
            count += inputs.size(0)
            if limit and count >= limit: break
            
    return np.concatenate(features), np.concatenate(labels_list)

from sklearn.svm import LinearSVC

def run_hybrid_model(train_loader, test_loader, device, save_dir='features'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print("VGG16 (Pretrained) dondurulmus katmanlarindan ozellikler cikariliyor...")
    vgg_base = models.vgg16(pretrained=True).to(device)
    
    print("Egitim seti ozellikleri cikariliyor (GAP: 512 boyutlu)...")
    X_train, y_train = extract_features(vgg_base, train_loader, device)
    print("Test seti ozellikleri cikariliyor...")
    X_test, y_test = extract_features(vgg_base, test_loader, device)
    
    # .npy dosyalarini kaydet
    train_feat_path = os.path.join(save_dir, 'X_train_features.npy')
    train_label_path = os.path.join(save_dir, 'y_train_labels.npy')
    test_feat_path = os.path.join(save_dir, 'X_test_features.npy')
    test_label_path = os.path.join(save_dir, 'y_test_labels.npy')

    np.save(train_feat_path, X_train)
    np.save(train_label_path, y_train)
    np.save(test_feat_path, X_test)
    np.save(test_label_path, y_test)
    
    # Print shapes
    print("\n" + "="*60)
    print("PHASE 4: EXTRACTED FEATURES (.npy) INFORMATION")
    print(f"X_train_features.npy shape: {X_train.shape}")
    print(f"y_train_labels.npy   shape: {y_train.shape}")
    print(f"X_test_features.npy  shape: {X_test.shape}")
    print(f"y_test_labels.npy    shape: {y_test.shape}")
    print("="*60 + "\n")
    
    print("LinearSVC egitiliyor (Buyuk veri seti icin optimize edildi)...")
    # SVC(kernel='linear') yerine buyuk veride cok daha hizli olan LinearSVC kullanıyoruz.
    clf = LinearSVC(dual=False, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    return acc, y_test, y_pred
