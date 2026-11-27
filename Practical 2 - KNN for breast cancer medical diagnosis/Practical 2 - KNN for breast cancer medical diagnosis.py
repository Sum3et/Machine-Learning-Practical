#IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#LOAD DATASET
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["Diagnosis"] = data.target

# Dataset target: 0 = malignant, 1 = benign
df["Diagnosis"] = df["Diagnosis"].map({
    0: "Malignant",
    1: "Benign"
})

print("FIRST FIVE ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

# Visualization
plt.figure(figsize=(8, 4))
sns.countplot(x="Diagnosis", data=df)
plt.title("Distribution of Benign and Malignant Tumors")
plt.show()

# STEP 2: EXPLORE DATA
print("\nDATASET INFORMATION:")
print(df.info())

print("\nMISSING VALUES:")
print(df.isnull().sum().sum())

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# Visualization: Diagnosis distribution
plt.figure(figsize=(6, 5))
df["Diagnosis"].value_counts().plot(
    kind="pie", autopct="%1.1f%%"
)
plt.title("Tumor Diagnosis Percentage")
plt.ylabel("")
plt.show()

#SELECT FEATURES AND TARGET
X = data.data
y = data.target

print("\nNUMBER OF FEATURES:", X.shape[1])
print("NUMBER OF RECORDS:", X.shape[0])

# Visualization: Feature relationship
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=df["mean radius"],
    y=df["mean texture"],
    hue=df["Diagnosis"]
)
plt.title("Mean Radius vs Mean Texture")
plt.show()

# STEP 4: SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING SAMPLES:", len(X_train))
print("TESTING SAMPLES:", len(X_test))

# Visualization
plt.figure(figsize=(6, 4))
plt.bar(
    ["Training Data", "Testing Data"],
    [len(X_train), len(X_test)]
)
plt.title("Training and Testing Data")
plt.ylabel("Number of Samples")
plt.show()

# STEP 5: FEATURE SCALING
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFEATURE SCALING COMPLETED")

# Visualization
plt.figure(figsize=(8, 4))
plt.boxplot(X_train[:, :5])
plt.title("Scaled Feature Distribution (First 5 Features)")
plt.xlabel("Feature Index")
plt.ylabel("Scaled Value")
plt.show()

# STEP 6: COMPARE DISTANCE METRICS
metrics = ["euclidean", "manhattan", "minkowski"]
results = {}

for metric in metrics:

    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric=metric
    )

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[metric] = accuracy

    print("\nDISTANCE METRIC:", metric)
    print("ACCURACY:", round(accuracy * 100, 2), "%")


# Visualization: Accuracy comparison
plt.figure(figsize=(8, 5))

plt.bar(
    results.keys(),
    [value * 100 for value in results.values()]
)

plt.title("KNN Accuracy Using Different Distance Metrics")
plt.xlabel("Distance Metric")
plt.ylabel("Accuracy (%)")
plt.ylim(80, 100)

for i, value in enumerate(results.values()):
    plt.text(i, value * 100 + 0.3,
             str(round(value * 100, 2)) + "%",
             ha="center")

plt.show()

# STEP 7: CONFUSION MATRIX
for metric in metrics:

    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric=metric
    )

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"]
    )

    plt.title("Confusion Matrix - " + metric)
    plt.xlabel("Predicted Diagnosis")
    plt.ylabel("Actual Diagnosis")
    plt.show()

# STEP 8: CLASSIFICATION REPORT
for metric in metrics:

    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric=metric
    )

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print("\nCLASSIFICATION REPORT:", metric)

    print(classification_report(
        y_test,
        y_pred,
        target_names=["Malignant", "Benign"]
    ))

# STEP 9: FINAL RESULT
print("\nFINAL ACCURACY COMPARISON")

for metric, accuracy in results.items():
    print(metric, ":", round(accuracy * 100, 2), "%")

best_metric = max(results, key=results.get)

print("\nBEST DISTANCE METRIC:", best_metric)
print("BEST ACCURACY:", round(results[best_metric] * 100, 2), "%")
