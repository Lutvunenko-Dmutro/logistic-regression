# -*- coding: utf-8 -*-   # вказує, що файл збережено у кодуванні UTF-8 (щоб підтримувалась українська мова)
"""
Система моніторингу завантаженості енергосистеми
Метод: Логістична регресія
Автор: Діма
"""

# === Імпорт бібліотек ===
import pandas as pd              # pandas – для роботи з таблицями (DataFrame)
import matplotlib.pyplot as plt  # matplotlib – для побудови графіків
import seaborn as sns            # seaborn – для красивих графіків і heatmap
from matplotlib.widgets import RadioButtons  # RadioButtons – інтерактивне меню у вікні matplotlib
from sklearn.model_selection import train_test_split  # train_test_split – поділ даних на train/test
from sklearn.linear_model import LogisticRegression   # LogisticRegression – модель логістичної регресії
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix
# roc_curve – для побудови ROC-кривої
# roc_auc_score – для обчислення AUC (площа під ROC-кривою)
# precision_recall_curve – для побудови PR-кривої
# confusion_matrix – для побудови матриці помилок

import statsmodels.api as sm     # statsmodels – для статистичних моделей (альтернативна логістична регресія)
import numpy as np               # numpy – для роботи з масивами та числовими обчисленнями

# === Стиль графіків ===
plt.style.use("seaborn-v0_8-darkgrid")  # встановлюємо стиль графіків (сітка + сучасний вигляд)

# === 1. Завантаження та підготовка даних ===
df = pd.read_csv("power_load_hourly.csv", parse_dates=["мітка_часу"])
# pd.read_csv – читає CSV-файл
# parse_dates=["мітка_часу"] – колонку "мітка_часу" перетворює у формат дати/часу

df["Overload"] = (df["навантаження_мвт"] > 0.9 * df["потужність_мвт"]).astype(int)
# створюємо нову колонку "Overload" (1 – перевантаження, 0 – норма)
# умова: якщо навантаження > 90% від потужності → 1, інакше 0
# .astype(int) – переводимо True/False у 1/0

df["hour"] = df["мітка_часу"].dt.hour
# додаємо колонку "hour" – година доби, витягнута з дати

# === Формування ознак і цілі ===
X = df[["навантаження_мвт", "температура_с", "вітер_м_с", "свято", "hour"]]
# X – матриця ознак (фактори, які впливають на перевантаження)
y = df["Overload"]  # y – цільова змінна (0 або 1)

# === 2. Поділ на train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
# train_test_split – ділить дані на навчальну і тестову вибірки
# test_size=0.25 – 25% даних під тест
# random_state=42 – фіксуємо випадковість для відтворюваності
# stratify=y – зберігаємо пропорцію класів (0/1)

# === 3. Модель ===
model = LogisticRegression(max_iter=500)  # створюємо модель логістичної регресії, max_iter=500 – кількість ітерацій
model.fit(X_train, y_train)               # навчаємо модель на train-даних
y_proba = model.predict_proba(X_test)[:, 1]  # прогнозуємо ймовірності класу 1 (перевантаження)
y_pred = model.predict(X_test)               # прогнозуємо класи (0 або 1)

# === 4. Метрики ===
fpr, tpr, _ = roc_curve(y_test, y_proba)          # fpr – false positive rate, tpr – true positive rate
auc = roc_auc_score(y_test, y_proba)              # AUC – площа під ROC-кривою
prec, rec, _ = precision_recall_curve(y_test, y_proba)  # prec – precision, rec – recall
cm = confusion_matrix(y_test, y_pred)             # матриця помилок

# === 5. Функції для графіків ===

def plot_roc(ax):
    ax.clear()  # очищаємо область перед малюванням
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}", color="#1f77b4", linewidth=2)  # малюємо ROC-криву
    ax.plot([0,1],[0,1],'k--')  # діагональ випадкового класифікатора
    ax.set_title("ROC-крива: якість прогнозу перевантаження", fontsize=14, fontweight="bold")
    ax.set_xlabel("1 - Specificity", fontsize=12)  # вісь X – помилкові спрацьовування
    ax.set_ylabel("Sensitivity", fontsize=12)      # вісь Y – чутливість
    ax.legend(fontsize=10, loc="lower right")      # легенда з AUC
    ax.grid(True, alpha=0.3)                       # сітка для зручності

def plot_pr(ax):
    ax.clear()
    ax.plot(rec, prec, color="#2ca02c", linewidth=2)  # малюємо PR-криву
    ax.set_title("Precision-Recall: баланс точності та повноти", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=12)     # вісь X – повнота
    ax.set_ylabel("Precision", fontsize=12)  # вісь Y – точність
    ax.grid(True, alpha=0.3)

def plot_confusion(ax):
    ax.clear()  # очищаємо область графіка перед малюванням
    sns.heatmap(
        cm,                      # матриця помилок (confusion matrix)
        annot=True,              # показати числа всередині клітинок
        fmt="d",                 # формат чисел – цілі (integer)
        cmap="Blues",            # кольорова схема (синя)
        ax=ax,                   # малюємо на переданій осі
        cbar=False,              # без кольорової шкали праворуч
        xticklabels=["Норма","Перевантаження"],  # підписи по осі X (прогноз)
        yticklabels=["Норма","Перевантаження"]   # підписи по осі Y (факт)
    )
    # heatmap – візуалізація матриці помилок
    ax.set_title("Матриця помилок класифікації", fontsize=14, fontweight="bold")  # заголовок
    ax.set_xlabel("Прогноз", fontsize=12)  # підпис осі X
    ax.set_ylabel("Факт", fontsize=12)     # підпис осі Y


def plot_logistic(ax):
    ax.clear()  # очищаємо область графіка
    # будуємо логістичну регресію тільки від однієї змінної – навантаження
    X_single = sm.add_constant(df["навантаження_мвт"])  
    logit_model = sm.Logit(df["Overload"], X_single).fit(disp=False)  # навчаємо модель

    # створюємо діапазон значень навантаження для побудови кривої
    x_vals = np.linspace(df["навантаження_мвт"].min(), df["навантаження_мвт"].max(), 200)
    X_plot = sm.add_constant(x_vals)  # додаємо константу
    y_pred_curve = logit_model.predict(X_plot)  # прогнозована ймовірність

    # довірчі інтервали для прогнозу
    pred = logit_model.get_prediction(X_plot)
    ci = pred.conf_int()  # межі інтервалів
    lower, upper = ci[:,0], ci[:,1]

    # малюємо дані та модель
    ax.scatter(df["навантаження_мвт"], df["Overload"], alpha=0.2, label="Дані", color="#1f77b4")  # точки даних
    ax.plot(x_vals, y_pred_curve, color="black", linewidth=2, label="Модель")  # крива логістичної регресії
    ax.fill_between(x_vals, lower, upper, color="gray", alpha=0.3, label="95% CI")  # довірчий інтервал
    ax.set_title("Логістична регресія: ймовірність перевантаження", fontsize=14, fontweight="bold")
    ax.set_xlabel("Навантаження, МВт", fontsize=12)
    ax.set_ylabel("Ймовірність", fontsize=12)
    ax.legend(fontsize=10, loc="lower right")  # легенда
    ax.grid(True, alpha=0.3)  # сітка


def plot_timeseries(ax):
    ax.clear()  # очищаємо область графіка
    # малюємо часовий ряд навантаження та генерації
    ax.plot(df["мітка_часу"], df["навантаження_мвт"], 
            label="Навантаження", color="#1f77b4", linewidth=2, alpha=0.9)
    ax.plot(df["мітка_часу"], df["потужність_мвт"], 
            label="Генерація", color="#2ca02c", linewidth=2, alpha=0.9)

    # виділяємо точки перевантаження червоним
    overload_points = df[df["Overload"] == 1]
    ax.scatter(overload_points["мітка_часу"], overload_points["навантаження_мвт"],
               color="#d62728", label="Перевантаження", s=30, alpha=0.8, edgecolor="k")

    ax.set_title("Часовий ряд: навантаження та генерація", fontsize=14, fontweight="bold")
    ax.set_xlabel("Час", fontsize=12)
    ax.set_ylabel("Потужність, МВт", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")  # легенда
    ax.grid(True, alpha=0.3)  # сітка


# === 6. Початкове вікно з меню ===
fig, ax = plt.subplots(figsize=(10,6))   # створюємо вікно з одним графіком
plt.subplots_adjust(left=0.28)           # залишаємо місце зліва під меню

plot_roc(ax)  # стартовий графік (ROC-крива)

# === 7. Меню (RadioButtons) ===
rax = plt.axes([0.02, 0.25, 0.2, 0.5])  # координати області для меню
radio = RadioButtons(
    rax,
    ('ROC-крива', 'Precision-Recall', 'Матриця помилок', 'Логістична регресія', 'Часовий ряд')
)

# функція, яка перемикає графіки
def update(label):
    if label == 'ROC-крива':
        plot_roc(ax)
    elif label == 'Precision-Recall':
        plot_pr(ax)
    elif label == 'Матриця помилок':
        plot_confusion(ax)
    elif label == 'Логістична регресія':
        plot_logistic(ax)
    elif label == 'Часовий ряд':
        plot_timeseries(ax)
    plt.draw()  # перемальовуємо графік

radio.on_clicked(update)  # прив’язуємо меню до функції update

plt.show()  # показуємо вікно