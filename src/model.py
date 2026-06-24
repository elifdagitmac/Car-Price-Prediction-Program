from sklearn.linear_model import LogisticRegression
import pandas as pd

# Block 1: Building and Training the Model
def train_model(X_train, y_train):
    # Logistic Regression model (It does classification)
    # We set max_iter=1000 so that the model has enough rounds to learn the data
    model = LogisticRegression(max_iter=1000)
    
    # Training process: It learns the boundary between the features and the 'Is it luxury?' target
    model.fit(X_train, y_train)
    return model

# Block 2: Making Predictions
def make_predictions(model, X_test):
    # It's not outputting the price anymore; it will return 0 (Economy) or 1 (Luxury)
    predictions = model.predict(X_test)
    return predictions
