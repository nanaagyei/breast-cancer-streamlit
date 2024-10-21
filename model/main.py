import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y  = data['diagnosis']

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train the model
    model = LogisticRegression() 
    model.fit(X_train, y_train)

    # test the model
    y_pred = model.predict(X_test)
    print(f"Accuracy of the model: {accuracy_score(y_test, y_pred)}")
    print(f"Classification report of the model:\n {classification_report(y_test, y_pred)}")

    return model, scaler


def get_clean_data():
    file_path = "../data/data.csv"
    data = pd.read_csv(file_path)
    data = data.drop(["id"], axis=1)

    data["diagnosis"] = data["diagnosis"].map({ 'M': 1, 'B': 0})

    return data


def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    # save the model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # save the scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)



if __name__ == "__main__":
    main()