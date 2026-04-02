import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        # PyTorch katman mimarisi    
        super(PyTorchMLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.Sigmoid())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_pytorch_model(model, X_train, y_train, epochs=1000, lr=0.1, l2_lambda=0.0):
    # PyTorch modeli eğitim döngüsü
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2_lambda)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()
        
    return model

def predict_pytorch(model, X):
    # test verisi uzerinden cikarim (inference) yapilmasi
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        outputs = model(X_t)
        return (outputs.numpy() > 0.5).astype(int)
