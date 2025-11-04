import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

# ====== АКТИВАЦИИ ======

def sigmoid(x): return 1 / (1 + np.exp(-x))
def dsigmoid(x): return sigmoid(x) * (1 - sigmoid(x))
def relu(x): return np.maximum(0, x)
def drelu(x): return (x > 0).astype(float)


# ====== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======

def generate_class_points(center, n=15, scale=0.5):
    return np.random.normal(loc=center, scale=scale, size=(n, 2))


def plot_epoch(X, y, W1, b1, W2, b2, epoch, loss, title_suffix="(MLP)"):
    """Отрисовка разделяющей границы для MLP."""
    plt.figure(figsize=(10, 5))

    # Создаём сетку точек для визуализации границы решений
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Прямой проход через сеть
    z1 = grid @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    probs = sigmoid(z2).reshape(xx.shape)

    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, cmap='RdBu')
    plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], c='blue', label='Класс 0')
    plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], c='red', label='Класс 1')

    plt.title(f"Эпоха {epoch} {title_suffix} — Потеря: {loss:.4f}")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend()
    plt.show()


# ====== ОБУЧЕНИЕ MLP ======

def train_mlp_interactive(X, y, hidden_neurons=5, eta=0.1, max_epochs=1000):
    """Интерактивное обучение многослойного персептрона (2 слоя)."""
    n_samples, n_features = X.shape
    n_output = 1

    # Инициализация весов
    W1 = np.random.randn(n_features, hidden_neurons) * 0.5
    b1 = np.zeros((1, hidden_neurons))
    W2 = np.random.randn(hidden_neurons, n_output) * 0.5
    b2 = np.zeros((1, n_output))

    for epoch in range(1, max_epochs + 1):
        # === Прямой проход ===
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        y_pred = sigmoid(z2)

        # === Потери (MSE) ===
        loss = np.mean((y - y_pred) ** 2)

        # === Обратное распространение ===
        dz2 = (y_pred - y) * dsigmoid(z2)
        dW2 = a1.T @ dz2 / n_samples
        db2 = np.mean(dz2, axis=0, keepdims=True)

        dz1 = (dz2 @ W2.T) * drelu(z1)
        dW1 = X.T @ dz1 / n_samples
        db1 = np.mean(dz1, axis=0, keepdims=True)

        # === Обновление весов ===
        W1 -= eta * dW1
        b1 -= eta * db1
        W2 -= eta * dW2
        b2 -= eta * db2

        # === Метрики ===
        preds_binary = (y_pred >= 0.5).astype(int)
        total_errors = np.sum(preds_binary != y)
        acc = (1 - total_errors / len(y)) * 100

        print(f"\nЭпоха {epoch}")
        print(f"Потеря: {loss:.4f}")
        print(f"Ошибок классификации: {total_errors}")
        print(f"Точность: {acc:.2f}%")
        print(f"W1.shape={W1.shape}, W2.shape={W2.shape}")

        plot_epoch(X, y, W1, b1, W2, b2, epoch, loss)

        user_input = input("Нажмите Enter для следующей эпохи или 'q' для выхода: ")
        if user_input.lower() == 'q':
            print("\nОбучение завершено.")
            break

    return W1, b1, W2, b2


# ====== ГЛАВНЫЙ ЦИКЛ ======

def one_experiment():
    print("\n===================================")
    print("   МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН (MLP)")
    print("===================================")

    c0x = float(input("\nВведите x центра Класса 0: "))
    c0y = float(input("Введите y центра Класса 0: "))
    c1x = float(input("Введите x центра Класса 1: "))
    c1y = float(input("Введите y центра Класса 1: "))
    n_points = int(input("\nВведите количество точек в каждом классе: "))

    center0 = np.array([c0x, c0y])
    center1 = np.array([c1x, c1y])

    np.random.seed(42)
    class0_points = generate_class_points(center0, n=n_points)
    class1_points = generate_class_points(center1, n=n_points)

    X = np.vstack((class0_points, class1_points))
    y = np.vstack((np.zeros((n_points, 1)), np.ones((n_points, 1))))  # бинарные метки 0/1

    hidden = int(input("\nВведите число нейронов в скрытом слое: "))
    eta = float(input("Введите скорость обучения η: "))

    train_mlp_interactive(X, y, hidden_neurons=hidden, eta=eta)


def main():
    while True:
        one_experiment()
        again = input("\nХотите повторить эксперимент? (y/n): ")
        if again.lower() != 'y':
            print("\nРабота программы завершена.")
            break


if __name__ == "__main__":
    main()
