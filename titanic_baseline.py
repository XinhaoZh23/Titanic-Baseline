import pandas as pd
import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import re
import copy

def extract_title(name):
    """从名字中提取称谓"""
    title = re.search(' ([A-Za-z]+)\.', name)
    if title:
        title = title.group(1)
    return title

def preprocess_title(title):
    """将称谓分组"""
    if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Mrs', 'Countess', 'Mme', 'Lady']:
        return 'Mrs'
    elif title in ['Miss', 'Ms', 'Mlle']:
        return 'Miss'
    elif title == 'Dr':
        return 'Dr'
    else:
        return 'Other'

def load_and_preprocess_data(file_path='train.csv', is_test=False):
    """加载和预处理数据"""
    print(f"\nLoading {'test' if is_test else 'training'} data...")
    df = pd.read_csv(file_path)
    
    if not is_test:
        print("\nFirst few rows of the dataset:")
        print(df.head())
    
    print("\nPreprocessing data...")
    
    # 1. 提取称谓特征
    df['Title'] = df['Name'].apply(extract_title)
    df['Title'] = df['Title'].apply(preprocess_title)
    title_dummies = pd.get_dummies(df['Title'], prefix='Title')
    df = pd.concat([df, title_dummies], axis=1)
    
    # 2. 处理缺失值
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
    df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
    
    # 3. 创建家庭规模特征
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 4. 处理分类特征
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # 5. 选择特征
    features = [
        'Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone',
        'Title_Mr', 'Title_Mrs', 'Title_Miss', 'Title_Dr', 'Title_Other'
    ]
    
    X = df[features].values
    
    if not is_test:
        y = df['Survived'].values
    else:
        y = None
        
    return (X, y, df['PassengerId'].values) if is_test else (X, y)

class TitanicModel(nn.Module):
    def __init__(self, input_size=11):
        super(TitanicModel, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.layer1 = nn.Linear(input_size, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.layer2 = nn.Linear(64, 32)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.relu(self.layer1(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.batch_norm3(x)
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer4(x))
        return x

class EarlyStopping:
    """早停策略"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
        return False

def train_with_cv(X, y, n_splits=5, epochs=150, lr=0.001, batch_size=32, patience=10):
    """使用交叉验证训练模型"""
    print("\n=== 开始交叉验证训练 ===")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    best_model = None
    best_score = 0
    best_scaler = None
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # 准备数据
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # 转换为张量
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
        # 初始化模型
        model = TitanicModel()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        early_stopping = EarlyStopping(patience=patience)
        
        # 训练
        train_data = torch.utils.data.TensorDataset(X_train, y_train.reshape(-1, 1))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val.reshape(-1, 1))
                val_pred = (val_outputs > 0.5).float()
                val_accuracy = (val_pred.view(-1) == y_val).float().mean()
                
                scheduler.step(val_loss)
                
                # 早停检查
                if early_stopping(model, val_loss):
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 30 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"Training Loss: {total_loss/len(train_loader):.4f}")
                print(f"Validation Loss: {val_loss.item():.4f}")
                print(f"Validation Accuracy: {val_accuracy.item():.4f}")
        
        # 加载最佳模型状态
        model.load_state_dict(early_stopping.best_model)
        
        # 评估当前折
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_pred = (val_outputs > 0.5).float()
            fold_accuracy = (val_pred.view(-1) == y_val).float().mean()
            print(f"\nFold {fold + 1} Accuracy: {fold_accuracy:.4f}")
            fold_scores.append(fold_accuracy.item())
            
            # 更新最佳模型
            if fold_accuracy > best_score:
                best_score = fold_accuracy
                best_model = copy.deepcopy(model)
                best_scaler = copy.deepcopy(scaler)
    
    print("\n=== 交叉验证结果 ===")
    print(f"平均准确率: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"最佳准确率: {best_score:.4f}")
    
    return best_model, best_scaler, best_score

class EnsembleModel:
    """集成模型，结合神经网络、随机森林和XGBoost"""
    def __init__(self):
        self.nn_model = None
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        # 标准化数据
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练神经网络
        print("\n训练神经网络...")
        self.nn_model, _, _ = train_with_cv(X, y)
        
        # 训练随机森林
        print("\n训练随机森林...")
        self.rf_model.fit(X_scaled, y)
        rf_score = accuracy_score(y, self.rf_model.predict(X_scaled))
        print(f"随机森林准确率: {rf_score:.4f}")
        
        # 训练XGBoost
        print("\n训练XGBoost...")
        self.xgb_model.fit(X_scaled, y)
        xgb_score = accuracy_score(y, self.xgb_model.predict(X_scaled))
        print(f"XGBoost准确率: {xgb_score:.4f}")
    
    def predict(self, X):
        # 标准化数据
        X_scaled = self.scaler.transform(X)
        
        # 神经网络预测
        X_torch = torch.FloatTensor(X_scaled)
        self.nn_model.eval()
        with torch.no_grad():
            nn_pred = (self.nn_model(X_torch) > 0.5).float().numpy()
        
        # 随机森林预测
        rf_pred = self.rf_model.predict(X_scaled)
        
        # XGBoost预测
        xgb_pred = self.xgb_model.predict(X_scaled)
        
        # 投票
        predictions = np.column_stack([nn_pred, rf_pred, xgb_pred])
        final_pred = np.mean(predictions, axis=1) >= 0.5
        return final_pred.astype(int)

def evaluate_metrics(y_true, y_pred):
    """计算并打印详细的评估指标"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n=== 详细评估指标 ===")
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"精确率 (Precision): {prec:.4f}")
    print(f"召回率 (Recall): {rec:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")
    print("\n混淆矩阵 (Confusion Matrix):")
    print("预测 0  预测 1")
    print(f"实际 0: {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"实际 1: {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    return acc

def evaluate_test_set(model, scaler, test_file='test.csv', submission_file='submission.csv'):
    print("\nEvaluating on test set...")
    
    # 加载和预处理测试数据
    X_test, _, test_ids = load_and_preprocess_data(test_file, is_test=True)
    print(f"Test set shape: {X_test.shape}")
    
    # 标准化特征
    X_test = scaler.transform(X_test)
    X_test = torch.FloatTensor(X_test)
    
    # 预测
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        predictions = (test_outputs > 0.5).float().numpy()
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions.flatten().astype(int)
    })
    submission_df.to_csv(submission_file, index=False)
    print(f"\nSubmission file created: {submission_file}")
    print("\nPreview of predictions:")
    print(submission_df.head())
    
    # 与gender_submission.csv比较
    gender_submission = pd.read_csv('gender_submission.csv')
    matching_predictions = (submission_df['Survived'] == gender_submission['Survived']).mean()
    print(f"\nAgreement with gender_submission baseline: {matching_predictions:.2%}")

def main():
    # 加载和预处理训练数据
    X, y = load_and_preprocess_data()
    
    # 创建和训练集成模型
    print("\n=== 训练集成模型 ===")
    ensemble = EnsembleModel()
    ensemble.fit(X, y)
    
    # 用集成模型进行最终验证集评估
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    val_pred = ensemble.predict(X_val)
    
    print("\n=== 最终验证集评估（集成模型）===")
    final_accuracy = evaluate_metrics(y_val, val_pred)
    print(f"\n最终验证集准确率: {final_accuracy:.4f}")
    
    # 评估测试集并创建提交文件
    print("\nEvaluating on test set...")
    X_test, _, test_ids = load_and_preprocess_data('test.csv', is_test=True)
    test_pred = ensemble.predict(X_test)
    
    # 创建提交文件
    submission_df = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': test_pred
    })
    submission_df.to_csv('submission.csv', index=False)
    print(f"\nSubmission file created: submission.csv")
    print("\nPreview of predictions:")
    print(submission_df.head())
    
    # 与gender_submission.csv比较
    gender_submission = pd.read_csv('gender_submission.csv')
    matching_predictions = (submission_df['Survived'] == gender_submission['Survived']).mean()
    print(f"\nAgreement with gender_submission baseline: {matching_predictions:.2%}")

if __name__ == "__main__":
    main()
