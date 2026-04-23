import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_training_results(histories, save_dir='plots'):
    """
    Eğitim ve test kayıp/doğruluk eğrilerini çizer.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 1. Kayıp (Loss) Eğrileri
    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        plt.plot(hist['train_loss'], '--', label=f'{name} (Train)')
        plt.plot(hist['test_loss'], '-', label=f'{name} (Test)')
    plt.title('Modellerin Kayıp (Loss) Eğrileri')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()

    # 2. Doğruluk (Accuracy) Eğrileri
    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        plt.plot(hist['train_acc'], '--', label=f'{name} (Train)')
        plt.plot(hist['test_acc'], '-', label=f'{name} (Test)')
    plt.title('Modellerin Doğruluk (Accuracy) Eğrileri')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_curves.png'))
    plt.close()

def plot_final_comparison(histories, hybrid_acc, save_dir='plots'):
    """
    Modellerin final başarılarını bar chart olarak kıyaslar.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12, 6))
    names = list(histories.keys()) + ['Hybrid (VGG+SVM)']
    accs = [histories[n]['test_acc'][-1] for n in histories.keys()] + [hybrid_acc * 100]
    
    colors = sns.color_palette('viridis', len(names))
    bars = plt.bar(names, accs, color=colors)
    
    plt.title('Final Model Başarı Oranları Karşılaştırması')
    plt.ylabel('Doğruluk (Accuracy %)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    
    # Değerleri bar üzerine yazma
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_comparison.png'))
    plt.close()

def save_confusion_matrix(y_true, y_pred, classes, model_name, save_dir='plots'):
    """
    Karmaşıklık matrisini çizer ve kaydeder.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_cm.png'))
    plt.close()
