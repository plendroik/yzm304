import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, ConfusionMatrixDisplay

if not os.path.exists("plots"):
    os.makedirs("plots")

def plot_learning_curves(history, title="Ogrenme Egrileri"):
    # kayip (loss) ve dogruluk (accuracy) egrilerinin cizimi
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history['train_loss'], label='Egitim Kaybi')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax1.plot(history['val_loss'], label='Dogrulama Kaybi')
    ax1.set_title('Kayıp Eğrisi')
    ax1.set_xlabel('Epok')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Egitim Dogrulugu')
    if 'val_acc' in history and len(history['val_acc']) > 0:
        ax2.plot(history['val_acc'], label='Dogrulama Dogrulugu')
    ax2.set_title('Dogruluk Eğrisi')
    ax2.set_xlabel('Epok')
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"plots/{title.replace(' ', '_').replace('/', '_')}.png")
    plt.close()

def evaluate_model(y_true, y_pred, model_name="Model"):
    # modellerin genel performans analizi (karisiklik matrisi)
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name}")
    plt.savefig(f"plots/{model_name.replace(' ', '_').replace('/', '_')}_cm.png")
    plt.close()
    
    return acc, rec, cm
