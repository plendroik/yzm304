import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.dataset import get_dataloaders, get_classes
from src.models import Model1_LeNet5, Model2_EnhancedLeNet5, get_model3_alexnet, get_model5_vgg16
from src.trainer import train_model, get_predictions
from src.hybrid import run_hybrid_model
from src.utils import plot_training_results, plot_final_comparison, save_confusion_matrix

def main():
    # GPU Ayari (En hızlı donanımı seçer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanilan Donanim: {device}")

    # PHASE 1: Dataset Setup & Preprocessing
    print("\n--- Phase 1: Veri Yukleme ve On Isleme ---")
    batch_size = 64
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    classes = get_classes()
    
    # Common Training Setup
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5
    histories = {}
    models_dict = {}

    # PHASE 2: Custom CNN Architectures (Model 1 & Model 2)
    print("\n--- Phase 2: Custom CNN Modelleri (Model 1 & 2) ---")
    
    # Model 1: Base Custom CNN
    print("\n[Model 1] Base Custom LeNet-5 Egitiliyor...")
    m1 = Model1_LeNet5().to(device)
    histories['Model 1'] = train_model(m1, train_loader, test_loader, criterion, 
                                     optim.Adam(m1.parameters(), lr=0.001), device, num_epochs)
    models_dict['Model 1'] = m1

    # Model 2: Regularized Custom CNN
    print("\n[Model 2] Regularized Custom LeNet-5 (BN+Dropout) Egitiliyor...")
    m2 = Model2_EnhancedLeNet5().to(device)
    histories['Model 2'] = train_model(m2, train_loader, test_loader, criterion, 
                                     optim.Adam(m2.parameters(), lr=0.001), device, num_epochs)
    models_dict['Model 2'] = m2

    # PHASE 3: Standard Literature CNN (Model 3)
    print("\n--- Phase 3: Literature CNN (Model 3) ---")
    print("\n[Model 3] AlexNet Egitiliyor...")
    m3 = get_model3_alexnet(pretrained=False).to(device)
    histories['Model 3'] = train_model(m3, train_loader, test_loader, criterion, 
                                     optim.Adam(m3.parameters(), lr=0.0001), device, num_epochs)
    models_dict['Model 3'] = m3

    # PHASE 4: Hybrid Machine Learning Model (Model 4)
    print("\n--- Phase 4: Hybrid Model (VGG16 Features + SVM) ---")
    hybrid_acc, y_true_hybrid, y_pred_hybrid = run_hybrid_model(train_loader, test_loader, device)
    print(f"Hybrid Model (Model 4) Doğruluğu: {hybrid_acc*100:.2f}%")

    # PHASE 5: Full Deep Learning CNN (Model 5) for Comparison
    print("\n--- Phase 5: Full CNN (Model 5) for Comparison ---")
    print("\n[Model 5] Full VGG16 Egitiliyor...")
    m5 = get_model5_vgg16(pretrained=True).to(device)
    histories['Model 5'] = train_model(m5, train_loader, test_loader, criterion, 
                                     optim.Adam(m5.parameters(), lr=0.0001), device, num_epochs)
    models_dict['Model 5'] = m5

    # Final Evaluation & Visualization
    print("\n--- Sonuclar Gorsellestiriliyor ---")
    
    # Confusion Matrices for all models
    for name, model in models_dict.items():
        y_true, y_pred = get_predictions(model, test_loader, device)
        save_confusion_matrix(y_true, y_pred, classes, name)
    
    save_confusion_matrix(y_true_hybrid, y_pred_hybrid, classes, "Model 4 (Hybrid)")

    # Performance Comparison Plots
    plot_training_results(histories)
    plot_final_comparison(histories, hybrid_acc)
    
    print("\nIslem tamamlandi. Grafiklere 'plots/' klasorunden ulasabilirsiniz.")

if __name__ == "__main__":
    main()
