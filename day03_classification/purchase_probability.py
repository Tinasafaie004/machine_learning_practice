import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.DataFrame({
    "age": [22, 25, 28, 35, 40, 45, 50, 55, 60, 65],
    "income": [30000, 35000, 40000, 50000, 60000,
               65000, 70000, 80000, 90000, 100000],
    "bought": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
})

X = df[["age", "income"]]
y = df["bought"]

X_train, X_test, y_train, y_test= train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
    #look inside y(orginial target) and make sure to preserve the proportions of the class when spliting
)

model= LogisticRegression()
model.fit(X_train, y_train)

predictions=model.predict(X_test)  #this will give us the final 0 and 1
print(predictions)

probabilities= model.predict_proba(X_test)
#we might get something like:
# [[0.80, 0.20],  0.80: class 0 / 0.20:class 1 --> so model.predit(..) returns 0
#  [0.25, 0.75],
#  [0.10, 0.90]]

#but what I only want the probability of class 1?
class_one_probs= model.predict_proba(X_test)[:,1]  #[:,1] --> :means for every row , column at index 1
print(class_one_probs)
print(y_test)

accuracy= accuracy_score(y_test, predictions)
print(accuracy)

cm=confusion_matrix(y_test, predictions)
print(cm)  #prints TN FP FN TP
