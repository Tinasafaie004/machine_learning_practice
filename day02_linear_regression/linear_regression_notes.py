# Lesson snippet: define X and y before running this example.
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#Creates an instance of the LinearRegression class.
model= LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
#first learn
model.fit(X_train, y_train)

#after learn we can also see what it has learned:
print(model.coef_)
print(model.intercept_)

#then use your knowledge
predictions=model.predict(X_test)

mae = mean_absolute_error (y_test, predictions)
