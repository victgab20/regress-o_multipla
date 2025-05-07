import pandas as pd
import matplotlib.pyplot as plt

base=pd.read_csv(r"C:\Users\victo\Downloads\financial_regression.csv")
df_numerico = base[["gold low", "gold high", "gold open", 
                           "oil close", "oil high", "oil low", "oil open", "gold close","CPI","nasdaq high"]]
df_numerico
novo = df_numerico.dropna()


# x = novo["nasdaq high"]
# y = novo["gold close"]
# media_x = x.mean()
# media_y = y.mean()

# print(novo)

# numerador = ((x - media_x) * (y - media_y)).sum()
# denominador = ((x - media_x).pow(2).sum() * (y - media_y).pow(2).sum()) ** 0.5

# r = numerador / denominador
# print(r)


df_reg = novo[['nasdaq high', 'gold close']]

x = df_reg['nasdaq high'].tolist()
y = df_reg['gold close'].tolist()

x_mean = 0
y_mean = 0
n = len(x)

for i in range(n):
    x_mean += x[i]
    y_mean += y[i]

x_mean /= n
y_mean /= n

numerador = 0
denominador = 0

for i in range(n):
    numerador += (x[i] - x_mean) * (y[i] - y_mean)
    denominador += (x[i] - x_mean) ** 2
beta1 = numerador / denominador

beta0 = y_mean - beta1 * x_mean

y_pred = [beta0 + beta1 * xi for xi in x]

equacao = f'y = {beta0:.2f} + {beta1:.2f} * x'
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Dados reais', alpha=0.5)
plt.plot(x, y_pred, color='red', label='Reta do OLS')
plt.xlabel('Nasdaq High')
plt.ylabel('Gold Close')
plt.title('Regressão Linear Simples - Nasdaq High vs Gold Close')
plt.legend()
plt.text(min(x), max(y), equacao, fontsize=12, color='red')

plt.show()

sse = ((df_reg["gold close"] -  y_pred)**2).sum()

sst = ((df_reg["gold close"] - y_mean)**2).sum()

r2 = 1 - (sse/sst)

print(r2)