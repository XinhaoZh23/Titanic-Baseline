from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from titanic_baseline import EnsembleModel, load_and_preprocess_data
import joblib
import os

app = Flask(__name__)

# 加载模型
def load_model():
    model_path = 'model/ensemble_model.joblib'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        # 如果模型不存在，训练一个新模型
        X, y = load_and_preprocess_data()
        model = EnsembleModel()
        model.fit(X, y)
        
        # 创建model目录（如果不存在）
        os.makedirs('model', exist_ok=True)
        # 保存模型
        joblib.dump(model, model_path)
    
    return model

# 全局变量存储模型
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取用户输入
        data = request.get_json()
        
        # 准备基础特征
        features = {
            'Pclass': int(data['pclass']),
            'Sex': data['sex'],
            'Age': float(data['age']),
            'SibSp': int(data['sibsp']),
            'Parch': int(data['parch']),
            'Fare': float(data['fare'])
        }
        
        # 创建DataFrame并进行预处理
        import pandas as pd
        df = pd.DataFrame([features])
        
        # 对类别特征进行编码
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        
        # 添加称谓特征（基于性别和年龄的简单规则）
        if df['Sex'].iloc[0] == 0:  # male
            if df['Age'].iloc[0] < 18:
                title = 'Other'
            else:
                title = 'Mr'
        else:  # female
            if df['Age'].iloc[0] < 18:
                title = 'Miss'
            else:
                title = 'Mrs'
                
        # 创建称谓哑变量
        df['Title_Mr'] = 1 if title == 'Mr' else 0
        df['Title_Mrs'] = 1 if title == 'Mrs' else 0
        df['Title_Miss'] = 1 if title == 'Miss' else 0
        df['Title_Dr'] = 0  # 假设没有医生
        df['Title_Other'] = 1 if title == 'Other' else 0
        
        # 添加家庭特征
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # 确保特征顺序与训练时一致
        feature_columns = [
            'Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone',
            'Title_Mr', 'Title_Mrs', 'Title_Miss', 'Title_Dr', 'Title_Other'
        ]
        
        # 预测
        prediction, probability = model.predict_proba(df[feature_columns])
        
        return jsonify({
            'survival_prediction': int(prediction[0]),
            'survival_probability': float(probability[0])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
