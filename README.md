# EDA_EXP_6

**Aim**

To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal.

**Algorithm**
Step 1 - Load the wine dataset and check its basic structure, shape, and missing values.

Step 2 - Plot univariate distributions for alcohol, volatile acidity, and pH to understand individual feature behavior.

Step 3 - Create bivariate boxplots to study relationships between wine quality and key predictors.

Step 4 - Compute and visualize the correlation matrix to identify feature relationships with wine quality.

Step 5 - Convert wine quality into a binary good/bad label for classification.

Step 6 - Split the dataset into training and testing sets for model evaluation.

Step 7 - Train a Logistic Regression model and predict wine quality on test data.

Step 8 - Evaluate the model using accuracy and confusion matrix, and detect outliers using boxplots.

**Program**
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
print(df.head())
print(df.shape)
print(df.dtypes)
```
```
features = ['alcohol','volatile acidity','pH']
summary = df[features].agg(['mean','median','std'])
print(summary)
```
```
plt.figure(figsize=(9,3))
for i, col in enumerate(features,1):
    plt.subplot(1,3,i)
    sns.histplot(df[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()
```
```
plt.figure(figsize=(8,3))
plt.subplot(1,2,1)
sns.scatterplot(x='alcohol', y='quality', data=df)
plt.subplot(1,2,2)
sns.scatterplot(x='volatile acidity', y='quality', data=df)
plt.show()
```
```
numeric_cols = df.select_dtypes(include=np.number).columns
outlier_counts = {}
for col in numeric_cols:
  Q1 = df[col].quantile(0.25)
  Q3 = df[col].quantile(0.75)
  IQR = Q3 - Q1
  lower = Q1 - 1.5*IQR
  upper = Q3 + 1.5*IQR
  outliers = df[(df[col] < lower) | (df[col] > upper)]
  outlier_counts[col] = len(outliers)
print(outlier_counts)
```
```
corr = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, cmap='coolwarm')
plt.show()
print(corr['quality'].sort_values(ascending=False))
```



**Output**

<img width="659" height="597" alt="image" src="https://github.com/user-attachments/assets/3ac20bd0-dd7e-4fbd-909b-ff28287b7706" />


<img width="728" height="288" alt="image" src="https://github.com/user-attachments/assets/c66616f4-d1bb-4539-95af-a8f70b7d47a0" />


<img width="630" height="441" alt="image" src="https://github.com/user-attachments/assets/4c5afa92-e765-4ec1-896e-431ebb30af5f" />


<img width="859" height="623" alt="image" src="https://github.com/user-attachments/assets/178a0a78-ef43-4d22-b0bf-69e1596fdf10" />


<img width="540" height="239" alt="image" src="https://github.com/user-attachments/assets/d04f403c-01b4-4966-b82c-2876299514f6" />

**Result**
Thus, To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal has successfully completed.
