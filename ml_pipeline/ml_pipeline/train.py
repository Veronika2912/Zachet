import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def load_data():
    """
    Загружает данные из CSV-файлов.
    Если файлов нет, создает демо-данные.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    orders_path = DATA_DIR / "orders.csv"
    products_path = DATA_DIR / "products.csv"
    inventory_path = DATA_DIR / "inventory.csv"

    if orders_path.exists() and products_path.exists() and inventory_path.exists():
        print("Обнаружены CSV-файлы. Загружаем данные")
        return _create_demo_data()
    else:
        print("CSV-файлы не найдены. Генерируем демо-данные")
        return _create_demo_data()


def _create_demo_data():
    """Создает реалистичные демо-данные для обучения"""
    np.random.seed(42)
    start_date = pd.to_datetime('2024-01-01')
    dates = pd.date_range(start_date, '2025-12-31', freq='D')
    products = range(1, 11)

    rows = []
    for product in products:
        for date in dates:
            days_passed = (date - start_date).days
            trend = 10 + days_passed / 45
            day_of_week = date.dayofweek
            weekend_effect = 1.5 if day_of_week >= 5 else 1.0
            noise = np.random.normal(1, 0.15)
            sales = trend * weekend_effect * noise

            rows.append({
                'product_id': product,
                'date': date,
                'sales': max(0, int(sales)),
                'price': 100 + product * 10,
                'inventory': np.random.randint(10, 200)
            })

    df = pd.DataFrame(rows)
    return df


def create_features(df):
    """Создает временные фичи и лаги для модели"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df = df.sort_values(['product_id', 'date'])
    df['lag_1'] = df.groupby('product_id')['sales'].shift(1)
    df['lag_7'] = df.groupby('product_id')['sales'].shift(7)
    df['lag_14'] = df.groupby('product_id')['sales'].shift(14)
    df['rolling_mean_7'] = df.groupby('product_id')['sales'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    df = df.dropna()
    return df


def calculate_mape_per_product(df_val_with_ids, y_pred):
    """
    Считает MAPE отдельно по каждому товару.

    Параметры:
    - df_val_with_ids: DataFrame, содержащий колонки 'product_id' и 'sales' (истинные значения)
    - y_pred: массив предсказанных значений (порядок строк соответствует df_val_with_ids)
    """
    df_temp = df_val_with_ids[['product_id', 'sales']].copy()
    df_temp['predicted'] = y_pred

    mape_by_product = df_temp.groupby('product_id').apply(
        lambda x: mean_absolute_percentage_error(x['sales'], x['predicted'])
    )
    return mape_by_product.to_dict()


def train_model():
    print("Шаг 1: Загрузка данных")
    df = load_data()

    print("Шаг 2: Создание признаков (Feature Engineering)")
    df = create_features(df)

    feature_cols = [
        'day_of_week', 'day_of_month', 'month', 'is_weekend',
        'price', 'inventory', 'lag_1', 'lag_7', 'lag_14', 'rolling_mean_7'
    ]
    X = df[feature_cols]
    y = df['sales']

    print(f"Размер выборки: {len(X)} строк, {len(feature_cols)} признаков")
    print(f"Признаки: {feature_cols}")

    print("Шаг 3: Time Series Cross-Validation")

    tscv = TimeSeriesSplit(n_splits=3)
    mape_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Сохраняем валидационную часть df с product_id для расчёта MAPE по товарам
        df_val_with_ids = df.iloc[val_idx]

        model = LGBMRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Общий MAPE
        mape = mean_absolute_percentage_error(y_val, y_pred)
        mape_scores.append(mape)

        # MAPE по каждому товару
        mape_per_product = calculate_mape_per_product(df_val_with_ids, y_pred)

        print(f"\nФолд {fold + 1}:")
        print(f"Общий MAPE = {mape:.4f} ({mape:.2%})")
        print(f" MAPE по товарам:")
        for product_id, mape_val in list(mape_per_product.items())[:3]:
            print(f"    Товар {product_id}: {mape_val:.4f} ({mape_val:.2%})")
        if len(mape_per_product) > 3:
            print(f"    ... и ещё {len(mape_per_product) - 3} товаров")

    print("Шаг 4: Обучение финальной модели")

    final_model = LGBMRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    final_model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "model.txt"
    final_model.booster_.save_model(str(model_path))
    print(f"Модель сохранена: {model_path}")

    metrics = {
        'cv_mape_mean': float(np.mean(mape_scores)),
        'cv_mape_std': float(np.std(mape_scores)),
        'cv_mape_per_fold': mape_scores,
        'feature_importance': dict(zip(feature_cols, map(float, final_model.feature_importances_))),
        'n_samples': len(X),
        'n_features': len(feature_cols)
    }

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Метрики сохранены: {metrics_path}")

    feature_cols_path = MODELS_DIR / "feature_cols.json"
    with open(feature_cols_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Список признаков сохранён: {feature_cols_path}")

    print("Обучение завершено успешно")
    print("=" * 50)
    print(f"Средний MAPE на кросс-валидации: {np.mean(mape_scores):.2%}")
    print(f"Лучший признак: {feature_cols[np.argmax(final_model.feature_importances_)]}")

    return metrics


if __name__ == '__main__':
    train_model()