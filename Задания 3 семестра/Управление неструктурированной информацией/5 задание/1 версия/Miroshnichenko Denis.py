import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')


# ====== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======

def generate_class_points(center, n=15, scale=0.5):
    """Создание точек вокруг заданного центра."""
    return np.random.normal(loc=center, scale=scale, size=(n, 2))


def plot_epoch(X, y, w, epoch, total_errors, title_suffix=""):
    """Отрисовка текущего состояния обучения."""
    plt.figure(figsize=(10, 5))

    x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    y_line = -(w[0] * x_line + w[2]) / w[1]

    preds = np.sign(np.dot(np.hstack((X, np.ones((X.shape[0], 1)))), w))
    misclassified = X[preds != y]

    # --- Левый график ---
    plt.subplot(1, 2, 1)
    plt.title(f"Эпоха {epoch} — Исходные данные {title_suffix}")
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', label='Класс 0 (истинный)')
    plt.scatter(X[y == 1, 0],  X[y == 1, 1],  color='red', label='Класс 1 (истинный)')
    plt.plot(x_line, y_line, 'g-', label='Разделяющая граница')
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend()

    # --- Правый график ---
    plt.subplot(1, 2, 2)
    plt.title(f"Эпоха {epoch} — Ошибки классификации {title_suffix}")
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', label='Класс 0')
    plt.scatter(X[y == 1, 0],  X[y == 1, 1],  color='red', label='Класс 1')
    if len(misclassified) > 0:
        plt.scatter(misclassified[:, 0], misclassified[:, 1],
                    color='black', marker='x', s=100, label='Ошибки')
    plt.plot(x_line, y_line, 'g-', label='Разделяющая граница')
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend()

    plt.suptitle(f"Всего ошибок: {total_errors}", fontsize=12)
    plt.tight_layout()
    plt.show()


# ====== КЛАССИЧЕСКИЙ ПЕРСЕПТРОН ======

def perceptron_train_interactive(X, y, eta=0.1, max_epochs=30):
    """Интерактивное обучение классического персептрона."""
    w = np.random.randn(3)
    X_bias = np.hstack((X, np.ones((X.shape[0], 1))))

    for epoch in range(1, max_epochs + 1):
        total_errors = 0
        sum_error = 0.0

        for i in range(len(X_bias)):
            f = np.dot(w, X_bias[i])
            if y[i] * f <= 0:
                error = y[i] - np.sign(f)
                w += eta * y[i] * X_bias[i]
                total_errors += 1
                sum_error += abs(error)

        accuracy = (1 - total_errors / len(X)) * 100

        print(f"\nЭпоха {epoch}")
        print(f"Ошибок классификации: {total_errors}")
        print(f"Невязка: {sum_error:.2f}")
        print(f"Точность: {accuracy:.2f}%")
        print(f"Параметры модели: w1={w[0]:.4f}, w2={w[1]:.4f}, w0={w[2]:.4f}")
        print(f"Уравнение границы: {w[0]:.4f}*x1 + {w[1]:.4f}*x2 + {w[2]:.4f} = 0")

        plot_epoch(X, y, w, epoch, total_errors, "(Персептрон)")

        user_input = input("Нажмите Enter для следующей эпохи или 'q' для выхода: ")
        if user_input.lower() == 'q' or total_errors == 0:
            print("\nОбучение завершено.")
            break

    return w


# ====== ГРАДИЕНТНЫЙ СПУСК (логистический персептрон) ======

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def perceptron_gradient_descent(X, y, eta=0.1, max_epochs=100):
    """Логистический персептрон с градиентным спуском."""
    w = np.random.randn(3)
    X_bias = np.hstack((X, np.ones((X.shape[0], 1))))

    # Преобразуем y из {-1, 1} в {0, 1} для сигмоиды
    y_bin = (y + 1) / 2

    for epoch in range(1, max_epochs + 1):
        z = np.dot(X_bias, w)
        y_pred = sigmoid(z)

        # Функция ошибки (MSE)
        loss = np.mean(0.5 * (y_bin - y_pred) ** 2)

        # Градиент по весам
        grad = np.dot((y_pred - y_bin) * y_pred * (1 - y_pred), X_bias) / len(X)
        w -= eta * grad

        # "Ошибки" условно — число точек, где предсказание < 0.5 для y=1 или >0.5 для y=0
        preds_binary = (y_pred >= 0.5).astype(int)
        preds_signed = preds_binary * 2 - 1
        total_errors = np.sum(preds_signed != y)

        print(f"\nЭпоха {epoch}")
        print(f"Ошибка (среднекв.): {loss:.4f}")
        print(f"Ошибок классификации: {total_errors}")
        print(f"Параметры модели: w1={w[0]:.4f}, w2={w[1]:.4f}, w0={w[2]:.4f}")
        print(f"Уравнение границы: {w[0]:.4f}*x1 + {w[1]:.4f}*x2 + {w[2]:.4f} = 0")

        plot_epoch(X, y, w, epoch, total_errors, "(Градиентный спуск)")

        nxt = input("Нажмите Enter для следующей эпохи или 'q' для выхода: ")
        if nxt.lower() == 'q':
            print("\nОбучение завершено.")
            break

    return w


# ====== ГЛАВНЫЙ ЦИКЛ ======

def one_experiment():
    print("\n===================================")
    print("   НАСТРОЙКА ЛИНЕЙНОГО ПЕРСЕПТРОНА")
    print("===================================")

    mode = int(input("\nВыберите режим обучения (1 — персептрон, 2 — градиентный спуск): "))

    print("\n1. ЗАДАНИЕ ЦЕНТРОВ КЛАССОВ:")
    c0x = float(input("Введите x-координату центра для Класса 0: "))
    c0y = float(input("Введите y-координату центра для Класса 0: "))
    c1x = float(input("Введите x-координату центра для Класса 1: "))
    c1y = float(input("Введите y-координату центра для Класса 1: "))

    n_points = int(input("\nВведите количество точек в каждом классе: "))

    center0 = np.array([c0x, c0y])
    center1 = np.array([c1x, c1y])

    np.random.seed(42)
    class0_points = generate_class_points(center0, n=n_points)
    class1_points = generate_class_points(center1, n=n_points)

    X = np.vstack((class0_points, class1_points))
    y = np.hstack((-np.ones(n_points), np.ones(n_points)))

    print("\nСгенерировано данных:")
    print(f"  Всего точек: {len(X)}")
    print(f"  Класс 0: {np.sum(y == -1)} точек (центр {center0})")
    print(f"  Класс 1: {np.sum(y == 1)} точек (центр {center1})")

    if mode == 1:
        perceptron_train_interactive(X, y)
    elif mode == 2:
        perceptron_gradient_descent(X, y)
    else:
        print("Некорректный выбор режима.")


def main():
    while True:
        one_experiment()
        again = input("\nХотите повторить эксперимент? (y/n): ")
        if again.lower() != 'y':
            print("\nРабота программы завершена.")
            break


if __name__ == "__main__":
    main()
