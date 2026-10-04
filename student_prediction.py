
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Create the dataset
data = {
    "Study_Hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 2, 3,
        5, 6, 7, 8, 9, 1, 4, 6, 8, 10,
        2, 5, 7, 9, 3, 4, 6, 8, 10, 1
    ],
    "Attendance": [
        40, 50, 60, 65, 70, 75, 80, 90, 45, 55,
        70, 80, 85, 95, 90, 35, 60, 75, 85, 100,
        50, 65, 80, 95, 55, 70, 90, 95, 100, 30
    ],
    "Passed": [
        0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
        0, 0, 1, 1, 0, 1, 1, 1, 1, 0
    ]
}

df = pd.DataFrame(data)

# 2. Separate features and target
X = df[["Study_Hours", "Attendance"]]
y = df["Passed"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("=== Student Performance Prediction ===")
print(f"Model accuracy: {accuracy * 100:.2f}%")

# 6. Predict a new student's result
study_hours = 6
attendance = 80

new_student = pd.DataFrame({
    "Study_Hours": [study_hours],
    "Attendance": [attendance]
})

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Prediction: Student is likely to pass.")
else:
    print("Prediction: Student may fail.")
