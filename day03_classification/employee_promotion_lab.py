import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.DataFrame({
    "years_experience": [
        1, 2, 2, 3, 4, 5, 6, 7, 8, 9,
        1, 3, 4, 5, 6, 7, 8, 10, 2, 9,
        3, 6, 4, 8, 10, 5, 7, 1, 9, 6
    ],

    "performance_score": [
        55, 60, 58, 65, 68, 72, 75, 80, 85, 90,
        50, 62, 70, 74, 78, 82, 88, 95, 57, 92,
        64, 76, 69, 86, 97, 73, 84, 52, 91, 79
    ],

    "projects_completed": [
        2, 3, 2, 4, 5, 5, 6, 7, 8, 9,
        1, 3, 4, 6, 6, 7, 8, 10, 2, 9,
        4, 5, 5, 8, 10, 6, 7, 2, 9, 6
    ],

    "promoted": [
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
        0, 1, 0, 1, 1, 1, 1, 0, 1, 1
    ]
})

X=df[["years_experience", "performance_score", "projects_completed"]]
y=df["promoted"]

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
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
