from sklearn.neural_network import MLPClassifier

def get_sklearn_mlp(hidden_layer_sizes=(10,), learning_rate_init=0.1, max_iter=1000, random_state=42, alpha=0.0):
    # scikit-learn kiyaslama modeli (benchmark) baslatimi
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='logistic',
        solver='sgd',
        alpha=alpha,
        batch_size=32,
        learning_rate_init=learning_rate_init,
        learning_rate='constant',
        max_iter=max_iter,
        random_state=random_state,
        momentum=0.0,
        early_stopping=False
    )
