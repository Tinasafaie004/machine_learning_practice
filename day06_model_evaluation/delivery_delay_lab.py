import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

rng = np.random.default_rng(18)
n = 1800

df = pd.DataFrame({
    "distance_km": rng.uniform(0.5, 18, n),
    "item_count": rng.integers(1, 9, n),
    "prep_minutes": rng.uniform(4, 35, n),
    "rush_hour": rng.integers(0, 2, n),
    "rain": rng.integers(0, 2, n),
    "courier_experience_months": rng.integers(1, 49, n),
    "receipt_code": rng.integers(100000, 999999, n),
})

risk = (
    -3.2
    + 0.09 * df["distance_km"]
    + 0.055 * df["prep_minutes"]
    + 0.11 * df["item_count"]
    + 0.65 * df["rush_hour"]
    + 0.45 * df["rain"]
    - 0.025 * df["courier_experience_months"]
    + 0.9 * (
        (df["distance_km"] > 10)
        & (df["rush_hour"] == 1)
    ).astype(int)
)

probability = 1 / (1 + np.exp(-risk))

df["late"] = (
    rng.random(n) < probability
).astype(int)

#classification
#late
#it was actually positive--late
#if it says it's going to be late how often is it actually late
#not sure but cause real life data is more complex and not based on formula??

X=df[["distance_km",
    "item_count",
    "prep_minutes",
    "rush_hour",
    "rain",
    "courier_experience_months"]]
y=df["late"]

X_dev,X_test,y_dev,y_test=train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

fold_plan=StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

shallow_tree=DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)



medium_tree=DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)

unrestricted_tree=DecisionTreeClassifier(
    max_depth=None,  #no amx depth for the tree
    random_state=42
)

unrestricted_forest=RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42
)

restricted_forest=RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    min_samples_leaf=8,
    random_state=42
)

candidates= {
    "Shallow tree": shallow_tree,
    "Medium tree": medium_tree,
    "Unrestricted tree": unrestricted_tree,
    "Unrestricted forest": unrestricted_forest,
    "Restricted forest": restricted_forest,
}

for name,candidate in candidates.items():
    print(f"\nCross Validation for {name}")
    candidate.fit(X_dev,y_dev)

    importances=pd.Series(
        candidate.feature_importances_,
        index=X_dev.columns
    ).sort_values(ascending=False)
    print(f"\nFeature importance for {name}")
    print(importances)

    results=cross_validate(
        candidate,
        X_dev,
        y_dev,
        cv=fold_plan,
        scoring={
            "recall":"recall",
            "precision":"precision",
            "f1":"f1"
        },
        return_train_score=True
    )

    print("Training recall:",results["train_recall"])
    print("Validation recall:", results["test_recall"])


    print("Mean training recall:", results["train_recall"].mean())
    print("Mean validation recall:", results["test_recall"].mean())
    print("Validation recall SD:", results["test_recall"].std())

    print("Mean validation precision:", results["test_precision"].mean())
    print("Mean validation F1:", results["test_f1"].mean())
    print("Recall gap:",results["train_recall"].mean()-results["test_recall"].mean())

#removed receipt_code from X
#now that I've chosen restricted forest it's time for final evaluation
restricted_forest_prediction= restricted_forest.predict(X_test)
print("\nRestricted Forest Final Evaluation:")
print("Test accuracy",accuracy_score(y_test, restricted_forest_prediction))
print("Confusion Matrix:\n",confusion_matrix(y_test,restricted_forest_prediction))
print("Classification Report:\n",classification_report(y_test,restricted_forest_prediction))
