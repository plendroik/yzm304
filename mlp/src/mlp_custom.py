import numpy as np

class CustomMLP:
    def __init__(self, layer_sizes, learning_rate=0.01, l2_lambda=0.0):
        # baslangic parametreleri ve agirlik atamalari
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.num_layers = len(layer_sizes) - 1
        self.best_n_steps_90_acc = -1 
        
        self.W = {}
        self.b = {}
        for i in range(1, self.num_layers + 1):
            scale = np.sqrt(1.0 / layer_sizes[i-1])
            self.W[i] = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * scale
            self.b[i] = np.zeros((layer_sizes[i], 1))
            
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def _sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))
    
    def _sigmoid_derivative(self, A):
        return A * (1 - A)

    def _compute_loss(self, A, Y):
        # ikili capraz entropi (bce) ve l2 kayip hesaplamasi
        m = Y.shape[0]
        A = np.clip(A, 1e-15, 1 - 1e-15)
        bce = -1/m * np.sum(Y.T * np.log(A) + (1 - Y).T * np.log(1 - A))
        
        l2_cost = 0
        if self.l2_lambda > 0:
            for i in range(1, self.num_layers + 1):
                l2_cost += np.sum(np.square(self.W[i]))
            l2_cost = (self.l2_lambda / (2 * m)) * l2_cost
            
        return bce + l2_cost

    def _forward(self, X):
        # ileri yayilim (forward propagation)
        self.A = {0: X.T}
        self.Z = {}
        for i in range(1, self.num_layers + 1):
            self.Z[i] = np.dot(self.W[i], self.A[i-1]) + self.b[i]
            self.A[i] = self._sigmoid(self.Z[i])
        return self.A[self.num_layers]

    def _backward(self, Y):
        # geri yayilim (backward propagation)
        m = Y.shape[0]
        self.dW = {}
        self.db = {}
        dZ = {}
        
        dZ[self.num_layers] = self.A[self.num_layers] - Y.T
        for i in range(self.num_layers, 0, -1):
            self.dW[i] = (1/m) * np.dot(dZ[i], self.A[i-1].T) + (self.l2_lambda / m) * self.W[i]
            self.db[i] = (1/m) * np.sum(dZ[i], axis=1, keepdims=True)
            if i > 1:
                dA_prev = np.dot(self.W[i].T, dZ[i])
                dZ[i-1] = dA_prev * self._sigmoid_derivative(self.A[i-1])

    def _update_params(self):
        # sgd ile agirliklarin guncellenmesi
        for i in range(1, self.num_layers + 1):
            self.W[i] -= self.learning_rate * self.dW[i]
            self.b[i] -= self.learning_rate * self.db[i]

    def predict(self, X):
        A_out = self._forward(X)
        return (A_out > 0.5).astype(int).T

    def evaluate_accuracy(self, X, Y):
        preds = self.predict(X)
        return np.mean(preds == Y)

    def train(self, X_train, Y_train, X_val, Y_val, epochs=1000, print_interval=100, batch_size=None):
        # egitim dongusu ve mini-batch
        m = X_train.shape[0]                            
        if batch_size is None or batch_size == 0:
            batch_size = m  
            
        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X_train[permutation, :]
            Y_shuffled = Y_train[permutation, :]
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size, :]
                Y_batch = Y_shuffled[i:i+batch_size, :]
                self._forward(X_batch)
                self._backward(Y_batch)
                self._update_params()
            
            # metriklerin hesaplanip kaydedilmesi
            A_out_full = self._forward(X_train)
            train_loss = self._compute_loss(A_out_full, Y_train)
            self.history['train_loss'].append(train_loss)
            
            val_out = self._forward(X_val)
            val_loss = self._compute_loss(val_out, Y_val)
            self.history['val_loss'].append(val_loss)
            
            train_acc = self.evaluate_accuracy(X_train, Y_train)
            val_acc = self.evaluate_accuracy(X_val, Y_val)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if val_acc >= 0.90 and self.best_n_steps_90_acc == -1:
                self.best_n_steps_90_acc = epoch
            
            if epoch % print_interval == 0 or epoch == epochs - 1:
                pass
