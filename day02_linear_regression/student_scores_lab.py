import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

scores= pd.read_csv("student_scores.csv")


model=LinearRegression()

X=scores[["study_hours", "attendance" , "assignments_completed"]]
y=scores["exam_score"]


X_train , X_test, y_train, y_test= train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model.fit(X_train, y_train)
predictions= model.predict(X_test)

mae=mean_absolute_error(y_test, predictions)

print(mae)
