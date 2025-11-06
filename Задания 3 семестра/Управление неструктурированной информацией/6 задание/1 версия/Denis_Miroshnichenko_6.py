import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')


# === АКТИВАЦИИ ===
def sigmoid(x): return 1 / (1 + np.exp(-x))
def dsigmoid(x): return sigmoid(x) * (1 - sigmoid(x))
def relu(x): return np.maximum(0, x)
def drelu(x): return (x > 0).astype(float)


# === ГЕНЕРАЦИЯ ДАННЫХ (XOR) ===
def generate_xor_data(n=100, noise=0.2):
    np.random.seed(42)
    X = np.random.rand(n, 2)
    y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(int)
    X += np.random.normal(0, noise, X.shape)
    return X, y.reshape(-1, 1)


# === ОБУЧЕНИЕ MLP ===
def train_mlp(X, y, hidden_neurons=6, eta=0.5, epochs=5000, verbose=False):
    n, d = X.shape
    W1 = np.random.randn(d, hidden_neurons) * 0.5
    b1 = np.zeros((1, hidden_neurons))
    W2 = np.random.randn(hidden_neurons, 1) * 0.5
    b2 = np.zeros((1, 1))

    for epoch in range(epochs):
        # === Прямой проход ===
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        y_pred = sigmoid(z2)

        # === Потери (MSE) ===
        loss = np.mean((y - y_pred) ** 2)

        # === Обратное распространение ===
        dz2 = (y_pred - y) * dsigmoid(z2)
        dW2 = a1.T @ dz2 / n
        db2 = np.mean(dz2, axis=0, keepdims=True)

        dz1 = (dz2 @ W2.T) * drelu(z1)
        dW1 = X.T @ dz1 / n
        db1 = np.mean(dz1, axis=0, keepdims=True)

        # === Обновление весов ===
        W1 -= eta * dW1
        b1 -= eta * db1
        W2 -= eta * dW2
        b2 -= eta * db2

        if verbose and epoch % 500 == 0:
            print(f"Эпоха {epoch}: Потеря = {loss:.4f}")

    return W1, b1, W2, b2, loss


# === ВИЗУАЛИЗАЦИЯ ===
def visualize_mlp(X, y, W1, b1, W2, b2):
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Слои
    z1 = grid @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_pred = sigmoid(z2).reshape(xx.shape)

    # Графики
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    titles = [
        "1. Исходные данные XOR",
        "2. Активации скрытого слоя (ReLU)",
        "3. Выход сети (Sigmoid)",
        "4. Решающее пространство MLP"
    ]

    # 1️⃣ Исходные данные
    axes[0].scatter(X[:, 0], X[:, 1],
                    c=['lightcoral' if yi else 'lightblue' for yi in y.flatten()],
                    edgecolors='k', s=70)
    axes[0].set_title(titles[0])
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[0].grid(True)

    # 2️⃣ Скрытый слой
    axes[1].imshow(a1[:, 0].reshape(xx.shape), extent=(x_min, x_max, y_min, y_max),
                   origin='lower', cmap='RdYlBu', alpha=0.7, aspect='auto')
    axes[1].set_title(titles[1])

    # 3️⃣ Выход сети
    axes[2].imshow(y_pred, extent=(x_min, x_max, y_min, y_max),
                   origin='lower', cmap='RdYlBu', alpha=0.7, aspect='auto')
    axes[2].set_title(titles[2])

    # 4️⃣ Решающее пространство
    axes[3].contourf(xx, yy, y_pred, levels=[0, 0.5, 1],
                     cmap='RdYlBu', alpha=0.6)
    axes[3].scatter(X[:, 0], X[:, 1],
                    c=['lightcoral' if yi else 'lightblue' for yi in y.flatten()],
                    edgecolors='k', s=70)
    axes[3].set_title(titles[3])
    axes[3].grid(True)

    for ax in axes:
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")

    plt.suptitle("Обучение многослойного персептрона",
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# === MAIN ===
def main():
    X, y = generate_xor_data(2000, noise=0.9)
    W1, b1, W2, b2, loss = train_mlp(X, y, hidden_neurons=6, eta=0.8, epochs=3000)
    visualize_mlp(X, y, W1, b1, W2, b2)
    print(f"\nФинальная потеря: {loss:.6f}")


if __name__ == "__main__":
    main()
