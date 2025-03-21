# main.py
from train_models import train_models

if __name__ == "__main__":
    results = train_models()
    print("Final Model Performance:")
    print(results)
