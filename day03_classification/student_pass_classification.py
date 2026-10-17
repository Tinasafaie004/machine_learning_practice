import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.DataFrame({
    "study_hours": [1, 2, 2, 3, 4, 5, 5, 6, 7, 8,
                    1, 3, 4, 6, 7, 9, 2, 5, 8, 9],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 82, 85, 90,
                   45, 62, 68, 78, 88, 95, 52, 72, 92, 97],
    "passed": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
               0, 0, 0, 1, 1, 1, 0, 1, 1, 1]
})

X=df[["study_hours", "attendance"]]
y=df["passed"]

model= LogisticRegression()

X_train, X_test, y_train , y_test=train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

predictions=model.predict(X_test)
print(predictions)

accuracy= accuracy_score(y_test, predictions)
print(accuracy)
precision=precision_score(y_test, predictions)
print(precision)
recall= recall_score(y_test, predictions)
print(recall)
f1= f1_score(y_test, predictions)
print(f1)
conf_matrix= confusion_matrix(y_test, predictions)
print(conf_matrix)
print(classification_report(y_test, predictions))
