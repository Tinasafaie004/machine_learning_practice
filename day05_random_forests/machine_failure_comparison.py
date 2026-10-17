import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

np.random.seed(42)
n=2500 #the number of machines
temperature=np.round(np.random.uniform(50,110,size=n),2)
vibration=np.round(np.random.uniform(0.5,10.0,size=n),2)
pressure=np.round(np.random.uniform(80,120,size=n),2)
machine_age= np.random.randint(1,21,size=n) #between 1 and 20
hours_since_service=np.random.randint(0,2001,size=n)
error_count=np.random.randint(0,11,size=n)  #between 0 and 10

df=pd.DataFrame({
    "temperature":temperature,
    "vibration":vibration,
    "pressure":pressure,
    "machine_age":machine_age,
    "hours_since_service":hours_since_service,
    "error_count":error_count
})

#checking
# print(df.head(3))
# print(df.shape)

#different params instead of just relying on age forexample
risk = (
    (temperature > 90).astype(int) +  #if you forget to convert to int , it will remain true false and it will do logical and kind of and only gets 0 or 1 --> there fore failure prob would remian 0.05 or 0.20
    (vibration > 7).astype(int) +
    (machine_age > 12).astype(int) +
    (hours_since_service > 1200).astype(int) +
    (error_count > 5).astype(int)
)

#now we need randomness cause maybe machine with over 4 risks stills survives
failure_probability= 0.05+(risk*0.15)

#this part is like a dice roll
random_numbers=np.random.random(n)  #2500 numbers between 0 and 1 to detemine which machine actually fails
failure=(
    random_numbers<failure_probability   #forexample if failure prob is 20%, since it's a unoform random number between 0 and 1, there's a 20% chance this will fail
).astype(int)  #because we want failure to be 0 and 1

df["failure"]=failure
# print(df.head(3))

failed_number=df["failure"].sum()
failed_percentage=df["failure"].mean()

print("Machines failed: ", failed_number)
print("Failure percentage",failed_percentage)
print(df.groupby("failure").mean())

#inside the second brackets of X, there is a python list --> then we turnm thsoe list of columns to data frame
X=df[["temperature",
    "vibration",
    "pressure",
    "machine_age",
    "hours_since_service",
    "error_count"]]

y=df["failure"]

X_train, X_test,y_train, y_test=train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model_tree=DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model_tree.fit(X_train,y_train)
tree_train_predictions=model_tree.predict(X_train)
tree_predictions=model_tree.predict(X_test)

print("Tree Model:")
print("Train accuracy",accuracy_score(y_train, tree_train_predictions))
print("Test accuracy",accuracy_score(y_test, tree_predictions))
print("Confusion Matrix:\n",confusion_matrix(y_test,tree_predictions))
print("Classification Report:\n",classification_report(y_test,tree_predictions))

# model_forest=RandomForestClassifier(
#     n_estimators=100,
#     max_depth=5,
#     random_state=42
# )

# model_forest.fit(X_train,y_train)
# forest_predictions=model_forest.predict(X_test)

# print("Forest Model:")
# print("Confusion Matrix:\n",confusion_matrix(y_test,forest_predictions))
# print("Classification Report:\n",classification_report(y_test,forest_predictions))

trees= [10,50,100,300]
for tree in trees:
    model_forest=RandomForestClassifier(
        n_estimators=tree,
        max_depth=5,
        random_state=42
    )

    model_forest.fit(X_train,y_train)

    depths=[estimator.get_depth() for estimator in model_forest.estimators_]

    forest_train_predictions=model_forest.predict(X_train)
    forest_predictions=model_forest.predict(X_test)

    print(f"Forest Model with {tree} trees ")
    print("Smallest depth:", min(depths))
    print("Largest depth:", max(depths))
    print("Train accuracy",accuracy_score(y_train, forest_train_predictions))
    print("Test accuracy",accuracy_score(y_test, forest_predictions))
    print("Confusion Matrix:\n",confusion_matrix(y_test,forest_predictions))
    print("Classification Report:\n",classification_report(y_test,forest_predictions))
    imporatnce=pd.DataFrame(  #if you don't add to a data frame this will be just numbers and you gott alook back at your X column to see which is which
        model_forest.feature_importances_,
        index=X.columns,
        columns=["importance"]
    )
    print("What was the most useful fetaure for this prediction?\n", imporatnce.sort_values(by="importance",ascending=False) )  #!!! there is an extra zero being printed


# MODEL ANALYSIS
# ## 1. Which model performed better:#    Decision Tree or Random Forest?
# ## 2. For machine failures, would you care more about#    precision or recall? Why?
# ## 3. Look at the confusion matrix.#    How many actual failures did your best model miss?
# ## 4. Did going from 100 → 300 trees meaningfully improve#    the model?
# ## 5. Would you choose 100 or 300 trees for production?#
#  Explain the tradeoff.
# ## 6. What evidence would make you worry that your model is overfitting?
