import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 避免Tkinter多线程报错，所有图片只保存不显示
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
import missingno as msno        
import concurrent.futures
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from scipy.stats import boxcox_llf
from sklearn.impute import SimpleImputer

# 控制台输出为UTF-8，解决中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)

def main():
    # ==================== 数据加载 ====================
    train = pd.read_csv("data/processed_train.csv")
    test = pd.read_csv("data/test.csv")
    print("数据加载完成")
    print(f"训练集: {train.shape}, 测试集: {test.shape}")

    # ==================== 探索性数据分析 ====================
    plt.figure(figsize=(30, 15))
    sns.boxplot(x=train["YearBuilt"], y=train["SalePrice"])
    plt.title('Box plots of year-to-year changes in house prices', fontsize=20)
    plt.xlabel('Year Built', fontsize=16)
    plt.ylabel('Sale Price', fontsize=16)
    plt.xticks(rotation=90)
    plt.savefig('year_price_boxplot.png', bbox_inches='tight')
    plt.close()
    print("生成图1: 年份与房价关系箱线图")

    plt.figure(figsize=(15, 8))
    sns.boxplot(x='OverallQual', y="SalePrice", data=train)
    plt.title('House Quality vs Sale Price', fontsize=16)
    plt.xlabel('Overall Quality', fontsize=12)
    plt.ylabel('Sale Price', fontsize=12)
    plt.ylim(0, 800000)
    plt.savefig('quality_price_boxplot.png', bbox_inches='tight')
    plt.close()
    print("生成图2: 房屋质量与房价关系箱线图")

    numeric_columns = train.select_dtypes(include=np.number).columns.drop('Id')
    corr = train[numeric_columns].corr().sort_values(by='SalePrice',ascending=False).round(2)
    print("\n房价相关性排序：")
    print(corr['SalePrice'].head(15))

    plt.figure(figsize=(15, 15))
    sns.heatmap(corr, vmax=.8, square=True)
    plt.title('Feature Correlation Heatmap', fontsize=20)
    plt.savefig('correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    print("生成图3: 全特征相关性热力图")

    plt.figure(figsize=(15, 8))
    cols = corr['SalePrice'].head(10).index
    cm = np.corrcoef(train[cols].values.T)
    sns.set(font_scale=1.0)
    sns.heatmap(cm, annot=True, yticklabels=cols.values, xticklabels=cols.values)
    plt.title('Top 10 Correlated Features with SalePrice', fontsize=16)
    plt.savefig('top_features_heatmap.png', bbox_inches='tight')
    plt.close()
    print("生成图4: 关键特征热力图")

    y = train.SalePrice
    print("\n房价统计描述:")
    print(y.describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True)
    plt.title('Sale Price Distribution', fontsize=16)
    plt.savefig('price_distribution.png', bbox_inches='tight')
    plt.close()
    print("生成房价分布图")

    print('偏度: %f' % y.skew())
    print('峰度: %f' % y.kurt())

    # ==================== 数据清洗 ====================
    plt.figure(figsize=(10, 6))
    plt.scatter(x=train.GrLivArea, y=train.SalePrice, alpha=0.6)
    plt.title('Living Area vs Sales Price (Original)', fontsize=14)
    plt.xlabel("GrLivArea", fontsize=12)
    plt.ylabel("SalePrice", fontsize=12)
    plt.ylim(0, 800000)
    plt.savefig('livarea_price_original.png', bbox_inches='tight')
    plt.close()
    print("生成图6: 原始居住面积与房价关系图")

    train.drop(train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 200000)].index, inplace=True)
    train.drop(train[(train["GarageYrBlt"] < 2000) & (train["SalePrice"] > 700000)].index, inplace=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(x=train.GrLivArea, y=train.SalePrice, alpha=0.6, color='green')
    plt.title('Living Area vs Sales Price (Cleaned)', fontsize=14)
    plt.xlabel("GrLivArea", fontsize=12)
    plt.ylabel("SalePrice", fontsize=12)
    plt.ylim(0, 800000)
    plt.savefig('livarea_price_cleaned.png', bbox_inches='tight')
    plt.close()
    print("生成图7: 清洗后居住面积与房价关系图")

    train_labels = train.SalePrice.values
    full = pd.concat([train.drop(['SalePrice', 'Id'], axis=1), test.drop('Id', axis=1)], ignore_index=True)
    print("\n合并后数据集形状:", full.shape)

    plt.figure(figsize=(16, 5))
    msno.matrix(full)
    plt.title('Missing Value Analysis', fontsize=16)
    plt.savefig('missing_values_matrix.png', bbox_inches='tight')
    plt.close()
    print("生成图8: 缺失值矩阵图")

    def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(5)
        print(f'数据集共有{df.shape[1]}列，其中{mis_val_table_ren_columns.shape[0]}列有缺失值')
        return mis_val_table_ren_columns

    print("\n缺失值分析:")
    print(missing_values_table(full))

    # Foster并行设计：划分任务 - 填充缺失值组1
    cols1 = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", 
             "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", 
             "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", 
             "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
    def fillna_none(col):
        full[col].fillna("None", inplace=True)
    print("\nFoster并行: 缺失值填充组1 (16项)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(fillna_none, cols1)

    cols2 = ["MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", 
             "Functional", "Electrical", "KitchenQual", "SaleType", 
             "Exterior1st", "Exterior2nd"]
    def fillna_mode(col):
        full[col].fillna(full[col].mode()[0], inplace=True)
    print("Foster并行: 缺失值填充组2 (10项)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(fillna_mode, cols2)

    cols3 = ["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", 
             "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
    def fillna_zero(col):
        full[col].fillna(0, inplace=True)
    print("Foster并行: 缺失值填充组3 (7项)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(fillna_zero, cols3)

    full['MSSubClass'] = full['MSSubClass'].astype(str)
    full['MSZoning'] = full.groupby('MSSubClass')['MSZoning'].transform(
        lambda x: x.fillna(x.mode()[0]))
    full['YrSold'] = full['YrSold'].astype(str)
    full['MoSold'] = full['MoSold'].astype(str)
    full['Functional'] = full['Functional'].fillna('Typ')
    full['Utilities'] = full['Utilities'].fillna('AllPub')
    full['KitchenQual'] = full['KitchenQual'].fillna("TA")

    NumStr = ["MSSubClass", "BedroomAbvGr", "KitchenAbvGr", "MoSold", 
              "YrSold", "YearBuilt", "YearRemodAdd", "LowQualFinSF", "GarageYrBlt"]
    def convert_to_str(col):
        full[col] = full[col].astype(str)
    print("\nFoster并行: 数值转字符串 (9项)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(convert_to_str, NumStr)

    full['TotalSF'] = full['1stFlrSF'] + full['2ndFlrSF'] + full['TotalBsmtSF']
    full['TotalBath'] = (full['BsmtFullBath'] + full['FullBath'] +
                         0.5 * (full['BsmtHalfBath'] + full['HalfBath']))
    full['TotalPorchSF'] = (full['OpenPorchSF'] + full['EnclosedPorch'] +
                            full['3SsnPorch'] + full['ScreenPorch'])

    y = train_labels
    lam_range = np.linspace(-2, 5, 100)
    llf = np.zeros(lam_range.shape, dtype=float)
    for i, lam in enumerate(lam_range):
        llf[i] = boxcox_llf(lam, y)
    lam_best = lam_range[llf.argmax()]
    print('\n最佳λ值: ', round(lam_best, 2))
    print('最大对数似然值: ', round(llf.max(), 2))
    plt.figure()
    plt.plot(lam_range, llf)
    plt.axvline(round(lam_best, 2), ls="--", color="r")
    plt.xlabel('λ', fontsize=12)
    plt.ylabel('对数似然值', fontsize=12)
    plt.title('Box-Cox变换优化', fontsize=14)
    plt.savefig('boxcox_optimization.png', bbox_inches='tight')
    plt.close()
    print("生成图9: Box-Cox变换优化图")

    full = pd.get_dummies(full)
    numeric_cols = full.select_dtypes(include=['int64', 'float64']).columns
    skew_vals = full[numeric_cols].apply(lambda x: skew(x.dropna()) if not x.empty else 0)
    skew_index = skew_vals[abs(skew_vals) >= 1].index
    full[skew_index] = np.log1p(full[skew_index])

    # ===== 新增：彻底填补所有NaN（避免PCA报错） =====
    imputer = SimpleImputer(strategy='mean')
    full = pd.DataFrame(imputer.fit_transform(full), columns=full.columns)
    print("全量缺失值填补完毕，剩余NaN数：", full.isnull().sum().sum())

    n_train = len(train)
    X = full.iloc[:n_train].values
    test_X = full.iloc[n_train:].values
    y = train_labels

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    y_log = np.log(y)
    test_X_scaled = scaler.transform(test_X)

    pca = PCA(n_components=min(150, X_scaled.shape[1]))
    X_scaled = pca.fit_transform(X_scaled)
    test_X_scaled = pca.transform(test_X_scaled)
    print(f"\nPCA降维后维度: {X_scaled.shape[1]}")

    def rmse_cv(model, X, y):
        rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
        return rmse

    models = [
        Lasso(alpha=0.0005, max_iter=10000),
        Ridge(alpha=10),
        BayesianRidge(),
        SVR(C=1.0, epsilon=0.1, kernel='rbf'),
        XGBRegressor(n_estimators=100, random_state=42),
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        RandomForestRegressor(n_estimators=100, random_state=42)
    ]
    names = ["Lasso", "Ridge", "Bay", "SVR", "Xgb", "GBR", "RF"]

    def train_and_score(args):
        name, model = args
        try:
            score = rmse_cv(model, X_scaled, y_log)
            result = f"{name}: {score.mean():.6f}, {score.std():.4f}"
            print(result)
            return (name, score.mean(), score.std())
        except Exception as e:
            print(f"模型 {name} 训练失败: {str(e)}")
            return (name, float('inf'), float('inf'))

    print("\nFoster并行模型训练与评估:")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        model_results = list(executor.map(train_and_score, zip(names, models)))

    class AverageWeight(BaseEstimator, RegressorMixin):
        def __init__(self, mod, weight):
            self.mod = mod
            self.weight = weight

        def fit(self, X, y):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.mod)) as executor:
                self.models_ = list(executor.map(lambda m: clone(m).fit(X, y), self.mod))
            return self

        def predict(self, X):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.models_)) as executor:
                preds = list(executor.map(lambda model: model.predict(X), self.models_))
            return np.average(np.array(preds), axis=0, weights=self.weight)

    best_models = [clone(model) for i, (name, model) in enumerate(zip(names, models)) if i in [0, 1, 2, 4]]
    weights = [0.25, 0.25, 0.25, 0.25]
    weighted_model = AverageWeight(mod=best_models, weight=weights)
    weighted_score = rmse_cv(weighted_model, X_scaled, y_log).mean()
    print(f"\nFoster并行加权模型得分: {weighted_score:.6f}")

    weighted_model.fit(X_scaled, y_log)
    preds = np.exp(weighted_model.predict(test_X_scaled))

    submission = pd.DataFrame({
        'Id': test.Id,
        'SalePrice': preds
    })
    submission.to_csv('submission.csv', index=False)
    print("\n提交文件已生成: submission.csv")
    print(f"预测房价范围: {preds.min():.2f} ~ {preds.max():.2f}")

if __name__ == '__main__':
    if sys.platform.startswith('win'):
        from multiprocessing import freeze_support
        freeze_support()
    main()