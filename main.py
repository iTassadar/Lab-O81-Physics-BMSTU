import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 0.5, 1, 1.5, 2])
y = np.array([0, -0.22, -0.44, -1.02, -1.02])

# Выполняем линейную регрессию для данных
coeff1 = np.polyfit(x, y, 1)

# Получаем прогнозируемые значения для x_smooth
x_smooth = np.linspace(min(x), max(x), 300)
y_pred1 = np.polyval(coeff1, x_smooth)

# Строим график с прогнозируемыми значениями
plt.minorticks_on()
plt.grid(which='major', color='#000000', linewidth=1)
plt.grid(which='minor', color='#000000', ls=':')
plt.plot(x_smooth, y_pred1, '-', label='Линейная регрессия', color='#00FF00')
plt.scatter(x, y, color='red', label='Экспериментальные точки', s=7)
plt.xlabel(r'$d, (см)$', fontsize=14)
plt.ylabel(r'$ln(T)$', fontsize=14)
plt.title('График зависимости')
plt.legend(loc='best')
plt.grid(True, linestyle='--', linewidth=1, color='grey', alpha=1)
plt.show()
