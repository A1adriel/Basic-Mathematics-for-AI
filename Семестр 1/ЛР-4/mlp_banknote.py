# Импорт библиотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

class MLP:
    """MLP с нуля"""
    
    def __init__(self, layers, learning_rate=0.01, l2_lambda=0.001):
        self.layers = layers  # [input_size, hidden1, ..., output_size]
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.weights = []
        self.biases = []
        self.activations = []
        
        # инициализация весов
        for i in range(len(layers) - 1):
            # Xavier init
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.activations = [X]
        
        # скрытые слои
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = self.relu(z)
            self.activations.append(a)
        
        # выход
        z_out = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        a_out = self.sigmoid(z_out)
        self.activations.append(a_out)
        
        return a_out
    
    def backward(self, X, y, output):
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        
        # ошибка на выходе
        error = output - y.reshape(-1, 1)
        deltas[-1] = error
        
        # backprop
        for i in range(len(self.weights) - 2, -1, -1):
            error = np.dot(deltas[i+1], self.weights[i+1].T) * self.relu_derivative(self.activations[i+1])
            deltas[i] = error
        
        # обновляем веса
        for i in range(len(self.weights)):
            # градиенты + L2
            dw = np.dot(self.activations[i].T, deltas[i]) / m + self.l2_lambda * self.weights[i]
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db
    
    def train(self, X, y, epochs, batch_size=32, X_val=None, y_val=None, verbose=True):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # перемешиваем
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # forward pass
                output = self.forward(X_batch)
                
                # лосс
                loss = -np.mean(y_batch * np.log(output + 1e-8) + (1 - y_batch) * np.log(1 - output + 1e-8))
                l2_loss = 0.5 * self.l2_lambda * sum(np.sum(w**2) for w in self.weights)
                total_loss = loss + l2_loss
                epoch_loss += total_loss
                
                # backward
                self.backward(X_batch, y_batch, output)
            

            avg_loss = epoch_loss / (len(X) // batch_size)
            train_losses.append(avg_loss)
            

            train_pred = self.predict(X)
            train_acc = accuracy_score(y, train_pred)
            train_accuracies.append(train_acc)
            

            if X_val is not None:
                val_pred_proba = self.forward(X_val)
                val_loss = -np.mean(y_val * np.log(val_pred_proba + 1e-8) + (1 - y_val) * np.log(1 - val_pred_proba + 1e-8))
                val_losses.append(val_loss)
                
                val_pred = self.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                val_accuracies.append(val_acc)
            
            if verbose and epoch % 50 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Train Acc = {train_acc:.4f}")
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def predict(self, X, threshold=0.5):
        proba = self.forward(X)
        return (proba >= threshold).astype(int).flatten()
    
    def predict_proba(self, X):
        return self.forward(X).flatten()

def find_optimal_threshold(model, X_val, y_val):
    """поиск оптимального порога (критерий Юдена)"""
    probas = model.predict_proba(X_val)
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.5
    best_j = 0
    
    for threshold in thresholds:
        predictions = (probas >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, predictions).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        j = tpr - fpr  # Критерий Юдена
        
        if j > best_j:
            best_j = j
            best_threshold = threshold
    
    return best_threshold, best_j

# загрузка данных
print("Загрузка данных Banknote Authentication...")

try:

    from sklearn.datasets import fetch_openml
    banknote = fetch_openml(name='banknote-authentication', version=1, as_frame=False)
    X = banknote.data
    y = banknote.target.astype(int)
    print("Данные загружены через fetch_openml")
    
except:

    print("Первый способ не сработал, пробуем альтернативный...")
    try:
        import pandas as pd
        import urllib.request
        import os
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
        
        if not os.path.exists('banknote_data.txt'):
            print("Скачивание данных...")
            urllib.request.urlretrieve(url, 'banknote_data.txt')
        
        data = pd.read_csv('banknote_data.txt', header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        print("Данные загружены через прямое скачивание")
        
    except:

        print("Используем встроенный датасет Breast Cancer...")
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = data.data
        y = data.target

print(f"Размерность данных: {X.shape}")
print(f"Количество классов: {len(np.unique(y))}")
print(f"Баланс классов: {np.bincount(y)}")

# делим на выборки
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42  # 0.25 * 0.8 = 0.2
)

print(f"\nРазмеры выборок:")
print(f"Обучающая: {X_train.shape}")
print(f"Валидационная: {X_val.shape}")
print(f"Тестовая: {X_test.shape}")

# нормализация
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# создаём модель
print("\nСоздание MLP...")

input_size = X_train.shape[1]
mlp = MLP(
    layers=[input_size, 64, 32, 1],
    learning_rate=0.01,
    l2_lambda=0.001
)

print("Начало обучения...")
train_losses, val_losses, train_acc, val_acc = mlp.train(
    X_train, y_train,
    epochs=300,
    batch_size=32,
    X_val=X_val,
    y_val=y_val,
    verbose=True
)

# ищем оптимальный порог
print("\nПоиск оптимального порога...")
optimal_threshold, j_score = find_optimal_threshold(mlp, X_val, y_val)
print(f"Оптимальный порог: {optimal_threshold:.4f}")
print(f"Критерий Юдена J: {j_score:.4f}")

# тест
print("\nОценка на тестовой выборке:")
test_predictions = mlp.predict(X_test, threshold=optimal_threshold)
test_accuracy = accuracy_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions)
test_auc = roc_auc_score(y_test, mlp.predict_proba(X_test))

print(f"Точность (Accuracy): {test_accuracy:.4f}")
print(f"F1-score: {test_f1:.4f}")
print(f"ROC-AUC: {test_auc:.4f}")
print(f"\nМатрица ошибок:")
print(confusion_matrix(y_test, test_predictions))
print(f"\nОтчет классификации:")
print(classification_report(y_test, test_predictions))

# графики
plt.figure(figsize=(15, 5))

# лосс
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Обучающая выборка')
if val_losses:
    plt.plot(val_losses, label='Валидационная выборка')
plt.title('Функция потерь')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)

# accuracy
plt.subplot(1, 3, 2)
plt.plot(train_acc, label='Обучающая выборка')
if val_acc:
    plt.plot(val_acc, label='Валидационная выборка')
plt.title('Точность')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# ROC
plt.subplot(1, 3, 3)
test_probas = mlp.predict_proba(X_test)
thresholds = np.linspace(0, 1, 100)
tprs = []
fprs = []
for threshold in thresholds:
    preds = (test_probas >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tprs.append(tpr)
    fprs.append(fpr)

plt.plot(fprs, tprs, 'b-', label='ROC кривая')
plt.plot([0, 1], [0, 1], 'r--', label='Случайный классификатор')
plt.scatter(fprs[np.argmax(thresholds == optimal_threshold)], 
           tprs[np.argmax(thresholds == optimal_threshold)], 
           color='red', s=100, label=f'Оптимальный порог ({optimal_threshold:.3f})')
plt.title('ROC кривая')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# инфо о модели
print("\n" + "="*50)
print("ИНФОРМАЦИЯ О МОДЕЛИ:")
print("="*50)
print(f"Архитектура: {mlp.layers}")
print(f"Количество параметров: {sum(w.size + b.size for w, b in zip(mlp.weights, mlp.biases))}")
print(f"Learning rate: {mlp.learning_rate}")
print(f"L2 регуляризация: {mlp.l2_lambda}")
print(f"Оптимальный порог τ: {optimal_threshold:.4f}")
print(f"Итоговая точность на тесте: {test_accuracy:.2%}")