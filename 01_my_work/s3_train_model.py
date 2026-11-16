import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def main():
    print("="*40)
    print("开始训练随机森林模型...")
    print("="*40)

    # 1. 加载数据 
    try:
        train_df = pd.read_csv("trail_work/01_my_work/s2_train_features.csv")
        test_df = pd.read_csv("trail_work/01_my_work/s2_test_features.csv")
    except FileNotFoundError:
        print("找不到特征文件，请检查路径或文件名！")
        return

    # 分离特征 (X) 和 标签 (y)
    X_train = train_df.drop(columns=['exp']).values
    y_train = train_df['exp'].values
    
    X_test = test_df.drop(columns=['exp']).values
    y_test = test_df['exp'].values

    # 2. 初始化模型 (n_estimators=?代表?棵树，n_jobs=-1代表用满所有CPU核心)
    model = RandomForestRegressor(n_estimators=100,
                                     n_jobs=-1,
                                       random_state=42)

    print("正在训练模型 (可能需要一小会儿)...")
    model.fit(X_train, y_train)

    # 3. 在测试集上进行预测
    print("正在评估模型表现...")
    y_pred = model.predict(X_test)

    # 4. 计算指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("-" * 30)
    print(f"模型训练完成！")
    print(f"评估指标 RMSE: {rmse:.4f} (越低越好，如果能低于1.0算是不错的结果)")
    print(f"评估指标 R2: {r2:.4f} (越接近1越好)")
    print("-" * 30)

    # 5. 保存模型文件，供后续 predict.py 使用
    model_filename = "trail_work/01_my_work/s3_lipophilicity_rf_model.pkl"
    joblib.dump(model, model_filename)
    print(f"模型已保存为: {model_filename}")

if __name__ == "__main__":
    main()