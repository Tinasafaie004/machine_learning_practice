import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)

n = 300

income = np.random.randint(25000, 130000, n)
credit_score = np.random.randint(500, 820, n)
debt = np.random.randint(2000, 35000, n)
account_age = np.random.randint(1, 13, n)

# Risk is influenced by MULTIPLE features
risk = (
    (credit_score < 620).astype(int) * 2
    + (debt > 22000).astype(int) * 2
    + (income < 45000).astype(int)
    + (account_age < 3).astype(int)
)

# Add some real-world randomness/noise
noise = np.random.binomial(1, 0.12, n)

loan_default = (risk >= 3).astype(int)

# Flip some labels so the data isn't perfectly predictable
loan_default = np.where(
    noise == 1,
    1 - loan_default,
    loan_default
)

data = pd.DataFrame({
    "income": income,
    "credit_score": credit_score,
    "debt": debt,
    "account_age": account_age,
    "loan_default": loan_default
})

X=data[["income", "credit_score", "debt", "account_age"]]
y=data["loan_default"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

depths=[1,2,3,4,5,None]  # None removes the depth limit; compare scores for evidence of overfitting

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f'Depth: {depth}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')


model_depth_five =DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model_depth_five.fit(X_train,y_train)
predictions_five=model_depth_five.predict(X_test)
print("Info for depth 5")
print(confusion_matrix(y_test, predictions_five))
print(classification_report(y_test, predictions_five))

model_depth_four =DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model_depth_four.fit(X_train,y_train)
predictions_four=model_depth_four.predict(X_test)
print("Info for depth 4")
print(confusion_matrix(y_test, predictions_four))
print(classification_report(y_test, predictions_four))
