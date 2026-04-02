import torch
import numpy as np
from src.dataset import load_and_preprocess_data
from src.mlp_custom import CustomMLP
from src.evaluate import plot_learning_curves, evaluate_model
from src.mlp_sklearn import get_sklearn_mlp
from src.mlp_pytorch import PyTorchMLP, train_pytorch_model, predict_pytorch

def inject_weights_to_pytorch(pt_model, custom_mlp):
    # agirliklarin pytorch modeline enjekte edilmesi ("ayni baslangic agirligi" kurali)
    with torch.no_grad():
        pt_model.network[0].weight.copy_(torch.tensor(custom_mlp.W[1], dtype=torch.float32))
        pt_model.network[0].bias.copy_(torch.tensor(custom_mlp.b[1].flatten(), dtype=torch.float32))
        pt_model.network[2].weight.copy_(torch.tensor(custom_mlp.W[2], dtype=torch.float32))
        pt_model.network[2].bias.copy_(torch.tensor(custom_mlp.b[2].flatten(), dtype=torch.float32))

def inject_weights_to_sklearn(sk_model, custom_mlp, X_train, y_train):
    # agirliklarin scikit-learn modeline enjekte edilmesi ("ayni baslangic agirligi" kurali)
    classes = np.unique(y_train)
    sk_model.partial_fit(X_train, y_train.flatten(), classes=classes)
    sk_model.coefs_[0] = custom_mlp.W[1].T.copy()
    sk_model.intercepts_[0] = custom_mlp.b[1].flatten().copy()
    sk_model.coefs_[1] = custom_mlp.W[2].T.copy()
    sk_model.intercepts_[1] = custom_mlp.b[2].flatten().copy()

def main():
    # veri yukleme ve on isleme
    data = load_and_preprocess_data("data/heart_disease.csv", test_size=0.2, val_size=0.1)  
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    input_dim = X_train.shape[1]
    
    # ozel temel mlp (1 gizli katman) egitimi
    base_mlp = CustomMLP([input_dim, 10, 1], learning_rate=0.1, l2_lambda=0.0)
    base_mlp_W_copy = base_mlp.W.copy()
    base_mlp_b_copy = base_mlp.b.copy()

    base_mlp.train(X_train, y_train, X_val, y_val, epochs=1000, print_interval=200, batch_size=32)
    evaluate_model(y_test.flatten(), base_mlp.predict(X_test).flatten(), "Temel_MLP")
    
    # ozel gelismis mlp (2 gizli katman + l2 regulerizasyonu) egitimi
    adv_mlp = CustomMLP([input_dim, 16, 8, 1], learning_rate=0.1, l2_lambda=0.01)
    adv_mlp.train(X_train, y_train, X_val, y_val, epochs=1000, print_interval=200, batch_size=32)
    evaluate_model(y_test.flatten(), adv_mlp.predict(X_test).flatten(), "Gelismis_MLP_L2")
    
    # accuracy - n_steps ikilemine gore secim kararinin hesaplanmasi
    if base_mlp.best_n_steps_90_acc != -1 and adv_mlp.best_n_steps_90_acc != -1:
        if base_mlp.best_n_steps_90_acc <= adv_mlp.best_n_steps_90_acc:
            print("Secim: Temel MLP daha erken 90% dogruluga ulasti.")
        else:
            print("Secim: Gelismis MLP daha erken 90% dogruluga ulasti.")
    
    plot_learning_curves(adv_mlp.history, title="Ogrenme_Egrileri_-_Gelismis_MLP")

    # scikit-learn kiyaslama dogrulamasi    
    sk_mlp = get_sklearn_mlp(hidden_layer_sizes=(10,), learning_rate_init=0.1, max_iter=1)
    temp_custom = CustomMLP([input_dim, 10, 1])
    temp_custom.W, temp_custom.b = base_mlp_W_copy, base_mlp_b_copy
    inject_weights_to_sklearn(sk_mlp, temp_custom, X_train, y_train)
    
    for ep in range(1000):
        sk_mlp.partial_fit(X_train, y_train.flatten())
        
    sk_preds = sk_mlp.predict(X_test)
    evaluate_model(y_test.flatten(), sk_preds, "Scikit-Learn_MLP")

    # pytorch kiyaslama dogrulamasi    
    pt_mlp = PyTorchMLP(input_size=input_dim, hidden_sizes=[10], output_size=1)
    inject_weights_to_pytorch(pt_mlp, temp_custom)
    pt_mlp = train_pytorch_model(pt_mlp, X_train, y_train, epochs=1000, lr=0.1, l2_lambda=0.0)
    pt_preds = predict_pytorch(pt_mlp, X_test)
    evaluate_model(y_test.flatten(), pt_preds.flatten(), "PyTorch_MLP")

if __name__ == "__main__":
    main()
