# Titanic Survival Prediction with PyTorch

A full-stack machine learning project that demonstrates end-to-end ML system development, from model training to production deployment.

## Technical Stack

### Machine Learning & Deep Learning
- **PyTorch**: Custom neural network implementation with advanced features (batch normalization, dropout)
- **Scikit-learn**: Data preprocessing, Random Forest implementation
- **XGBoost**: Gradient boosting implementation
- **Ensemble Learning**: Custom voting classifier combining multiple models
- **Cross Validation**: K-fold validation for robust model evaluation

### Backend Development
- **Flask**: RESTful API development, web server implementation
- **Pandas & NumPy**: Data manipulation and numerical computations
- **Joblib**: Model serialization and persistence

### Frontend Development
- **HTML5/CSS3**: Responsive web interface
- **JavaScript**: Asynchronous API calls, dynamic UI updates
- **Bootstrap 5**: Modern and responsive design
- **AJAX**: Real-time prediction without page reload

### DevOps & Deployment
- **Docker**: Application containerization
- **Docker Compose**: Container orchestration
- **Gunicorn**: Production-grade WSGI server
- **Environment Management**: Production/development configurations

## Features

- Advanced feature engineering (title extraction, family size, etc.)
- Ensemble model combining multiple algorithms:
  - Neural Network (PyTorch)
  - Random Forest
  - XGBoost
- Cross-validation and early stopping
- Web interface using Flask
- Docker containerization for easy deployment

## Setup

### Option 1: Local Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download the Titanic dataset:
   - Go to https://www.kaggle.com/c/titanic/data
   - Download `train.csv`
   - Place `train.csv` in the project root directory

3. Run the Flask application:
```bash
python app.py
```

### Option 2: Docker Deployment

1. Build and run with Docker:
```bash
docker build -t titanic-predictor .
docker run -p 5000:5000 titanic-predictor
```

Or using Docker Compose:
```bash
docker-compose up
```

## Model Architecture

### Neural Network Component
- Input layer (11 features)
- Hidden layers with batch normalization and dropout
- Output layer with sigmoid activation

### Features Used
- Passenger Class (Pclass)
- Sex
- Age
- Fare
- Family Size
- Title (extracted from Name)
- Sibling/Spouse Count (SibSp)
- Parent/Child Count (Parch)

## Web Interface

Access the prediction interface at `http://localhost:5000` after starting the application. The interface allows you to:
- Input passenger details
- Get survival predictions
- View prediction probabilities

## Model Training

To train a new model:
```bash
python titanic_baseline.py
```

## Project Structure
```
├── app.py                 # Flask web application
├── titanic_baseline.py    # Model training and evaluation
├── templates/            
│   └── index.html        # Web interface template
├── static/               
│   ├── style.css         # CSS styles
│   └── script.js         # Frontend JavaScript
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
└── model/               # Saved model files
```
