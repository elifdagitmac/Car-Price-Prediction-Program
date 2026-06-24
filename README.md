# Car Price Classifier

A machine learning project that classifies cars as **Economy (0)** or **Luxury (1)** 
based on technical specifications, using Logistic Regression.

## Results
| Metric | Score |
|---|---|
| Accuracy | 80% |
| Luxury recall | 94% |
| Economy recall | 87% |

## Project Structure
Car-Price-Prediction-Program/

├── src/

│   ├── data_prep.py      # Data cleaning & encoding

│   ├── model.py          # Model training & prediction

│   └── utils.py          # Evaluation & visualization

├── models/

│   └── car_price_model.pkl

├── car_data.csv

├── main.py

└── requirements.txt

## Features Used
| Feature | Description |
|---|---|
| `fueltype` | Fuel type (gas / diesel) |
| `carbody` | Body style |
| `horsepower` | Engine power |
| `enginesize` | Engine displacement |
| `is_luxury` | Target variable (derived from price median) |

## Visualizations
- Confusion matrix
- Correlation heatmap
- Feature importance (logistic regression coefficients)
- Sigmoid curve with decision boundary

## Getting Started
```bash
pip install -r requirements.txt
python main.py
```

## Tech Stack
- Python 3
- scikit-learn (Logistic Regression)
- pandas, matplotlib, seaborn
