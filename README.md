# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


df = pd.read_csv('Placement_Data.csv')


df1 = df.copy()


df1 = df1.drop(['sl_no', 'salary'], axis=1)


print("Null values:\n", df1.isnull().sum())
print("\nDuplicate rows:", df1.duplicated().sum())


le = LabelEncoder()
df1['gender'] = le.fit_transform(df1['gender'])
df1['ssc_b'] = le.fit_transform(df1['ssc_b'])
df1['hsc_b'] = le.fit_transform(df1['hsc_b'])
df1['hsc_s'] = le.fit_transform(df1['hsc_s'])
df1['degree_t'] = le.fit_transform(df1['degree_t'])
df1['workex'] = le.fit_transform(df1['workex'])
df1['specialisation'] = le.fit_transform(df1['specialisation'])
df1['status'] = le.fit_transform(df1['status'])


x = df1.drop('status', axis=1)
y = df1['status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("\nConfusion Matrix:\n", confusion)
print("\nClassification Report:\n", report)


cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['Not Placed', 'Placed'])
cm_display.plot()

```

## Output:
<img width="1392" height="697" alt="Screenshot 2026-02-12 135544" src="https://github.com/user-attachments/assets/dd85a2b9-788a-4da1-b573-5b2802c91234" />
<img width="1400" height="604" alt="Screenshot 2026-02-12 135600" src="https://github.com/user-attachments/assets/e5ad1eb7-c038-4055-9242-ce06b5c7569f" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
