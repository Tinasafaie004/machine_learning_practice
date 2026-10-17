# Lesson snippet: define X and y before running this example.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

X_dev,X_test, y_dev,y_test=train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y  #this will make sure the proportions are approximately the same
)

fold_plan=StratifiedKFold(
    n_splits=5,
    shuffle=True,   #shuffles the rows before forming the fold
    random_state=42
)

candidate=RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    min_samples_leaf=5,  #each leaf must conatin at least 5 training rows
    #
    random_state=42
)

#we can see how ell the firest receipe works
#how about high variance?
results=cross_validate(
    candidate,
    X_dev,
    y_dev,
    cv=fold_plan,  #what splitting plan
    scoring={  #name: metric
        "catching_delays":"recall",   #you would retrieve the recall scores using: results["test_catching_delays"]
        "reliable_alerts":"precision", #results["test_reliable_alerts"]
        "balance":"f1"
    },
    return_train_score=True
)

print("Training recall:",results["train_catching_delays"]) #fold scored inside cross validation not the entire train_recall
print("Validation recall:", results["test_catching_delays"])


print("Mean training recall:", results["train_catching_delays"].mean())
print("Mean validation recall:", results["test_catching_delays"].mean())
print("Validation recall SD:", results["test_catching_delays"].std())

print("Mean validation precision:", results["test_reliable_alerts"].mean())
print("Mean validation F1:", results["test_balance"].mean())
