document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // 收集表单数据
    const formData = {
        pclass: document.getElementById('pclass').value,
        sex: document.getElementById('sex').value,
        age: document.getElementById('age').value,
        fare: document.getElementById('fare').value,
        sibsp: document.getElementById('sibsp').value,
        parch: document.getElementById('parch').value,
        embarked: document.getElementById('embarked').value
    };

    try {
        // 发送预测请求
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (response.ok) {
            // 显示预测结果
            const resultDiv = document.getElementById('result');
            const predictionText = document.getElementById('prediction-text');
            const probabilityBar = document.getElementById('probability-bar');
            const alertDiv = resultDiv.querySelector('.alert');

            // 更新UI
            resultDiv.style.display = 'block';
            
            if (result.survival_prediction === 1) {
                alertDiv.className = 'alert alert-success';
                predictionText.textContent = 'This passenger would likely SURVIVE!';
            } else {
                alertDiv.className = 'alert alert-danger';
                predictionText.textContent = 'This passenger would likely NOT SURVIVE.';
            }

            // 更新概率条
            const probability = result.survival_probability * 100;
            probabilityBar.style.width = `${probability}%`;
            probabilityBar.textContent = `${probability.toFixed(1)}%`;
            
            // 滚动到结果
            resultDiv.scrollIntoView({ behavior: 'smooth' });
        } else {
            throw new Error(result.error || 'Prediction failed');
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
});
