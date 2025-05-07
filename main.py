import pandas as pd
import matplotlib.pyplot as plt
import random

base = pd.read_csv(r"C:\Users\victo\Downloads\financial_regression.csv")

print(base.columns)

df_numerico = base[[
    "gold low", "gold high", "gold open", 
    "oil close", "oil high", "oil low", "oil open", 
    "gold close", "CPI", "nasdaq high"
]].dropna()

y = df_numerico["gold close"].tolist()
X = df_numerico.drop(columns=["gold close"]).values.tolist()

#this little shit serve to stardarzine using normalização Z-score

def standardize(X):
    n_cols = len(X[0])
    means = [sum(row[i] for row in X) / len(X) for i in range(n_cols)]
    stds = []
    for i in range(n_cols):
        variance = sum((row[i] - means[i]) ** 2 for row in X) / len(X)
        stds.append(variance ** 0.5)
    
    X_scaled = []
    for row in X:
        scaled_row = [(row[i] - means[i]) / stds[i] if stds[i] != 0 else 0 for i in range(n_cols)]
        X_scaled.append([1.0] + scaled_row)
    return X_scaled

X_scaled = standardize(X)

#calcula o custo computacional, com fórmula lá explicada:

def compute_cost(X, y, theta):
    m = len(y)
    total = 0
    for i in range(m):
        prediction = sum(X[i][j] * theta[j] for j in range(len(theta)))
        total += (prediction - y[i]) ** 2
    return total / (2 * m)

#gradient descendente implementado com o learning rate usando a técnica de learning rate decay

def gradient_descent(X, y, theta, alpha_inicial, iterations, decay=0.001):
    m = len(y)
    cost_history = []

    for t in range(iterations):
        alpha = alpha_inicial / (1 + decay * t)

        gradients = [0] * len(theta)

        for i in range(m):
            prediction = sum(X[i][j] * theta[j] for j in range(len(theta)))
            error = prediction - y[i]
            for j in range(len(theta)):
                gradients[j] += error * X[i][j]

        for j in range(len(theta)):
            theta[j] -= alpha * gradients[j] / m

        cost_history.append(compute_cost(X, y, theta))
        print(alpha)

    return theta, cost_history



def mean_absolute_error(y_true, y_pred):
    return sum(abs(y_true[i] - y_pred[i]) for i in range(len(y_true))) / len(y_true)

def r2_score(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((val - mean_y) ** 2 for val in y_true)
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true)))
    return 1 - ss_res / ss_tot


#lógica do k-fold, usando apenas essas seed para reproduzir, 

def manual_kfold(X, y, k=5, seed=42):
    random.seed(seed)
    indices = list(range(len(X)))
    random.shuffle(indices)
    fold_sizes = [len(X) // k] * k
    for i in range(len(X) % k):
        fold_sizes[i] += 1
    folds = []
    current = 0
    for size in fold_sizes:
        test_indices = indices[current:current+size]
        train_indices = indices[:current] + indices[current+size:]
        folds.append((train_indices, test_indices))
        current += size
    return folds

folds = manual_kfold(X_scaled, y, k=5)

r2_scores = []

mae_scores = []

#validação cruzada K-Fold para avaliar o modelo de regressão linear com duas métricas

for train_idx, test_idx in folds:
    X_train = [X_scaled[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X_scaled[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    theta = [0.0] * len(X_train[0])
    theta, _ = gradient_descent(X_train, y_train, theta, alpha_inicial=0.01, iterations=1000, decay=0.001)

    y_pred = [sum(X_test[i][j] * theta[j] for j in range(len(theta))) for i in range(len(X_test))]
    
    r2_scores.append(r2_score(y_test, y_pred))
    mae_scores.append(mean_absolute_error(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Gold Close (Real)")
plt.ylabel("Gold Close (Previsto)")
plt.title("Scatterplot - Reais vs Previsto (último fold)")
min_y, max_y = min(y_test + y_pred), max(y_test + y_pred)
plt.plot([min_y, max_y], [min_y, max_y], color='red', linestyle='--')
plt.grid(True)
plt.show()

print("R² médio com K-Fold:", sum(r2_scores) / len(r2_scores))
print("MAE médio com K-Fold:", sum(mae_scores) / len(mae_scores))

