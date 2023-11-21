import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Ваши данные
x = np.array([20, 40, 60, 80, 100])
y = np.array([0.00, -0.13, -0.13, -0.13, -0.94])

# Преобразуем x в двумерный массив, так как LinearRegression требует двумерный вход
X = x.reshape(-1, 1)

# Создаем и обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X, y)

# Получаем предсказанные значения
y_pred = model.predict(X)

# Получаем прогнозируемые значения для x_smooth
x_smooth = np.linspace(min(x), max(x), 300)
y_pred = model.predict(x_smooth.reshape(-1, 1))

# Строим график
plt.minorticks_on()
plt.grid(which='major', color='#000000', linewidth=1)
plt.grid(which='minor', color='#000000', ls=':')
plt.plot(x_smooth, y_pred, '-', label='Метод Наименьших Квадратов (scikit-learn)', color='#00FF00')
plt.scatter(x, y, color='red', label='Экспериментальные точки', s=30, marker='o')
plt.xlabel('d, cm', fontsize=14)
plt.ylabel(r'$\ln\left(\frac{I_1}{I_n}\right)$', fontsize=14)
plt.title('График зависимости')
plt.legend(loc='best')
plt.grid(True, linestyle='--', linewidth=1, color='grey', alpha=1)
plt.show()
