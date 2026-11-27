#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
print("All libraries imported successfully!")


#LOAD THE DATASET
df = pd.read_csv("palmerpenguins.csv")

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)
print("\nFirst 10 Records:")
print(df.head(10))
print("\nColumn Names:")
print(df.columns.tolist())


# VISUALIZATION 1: BILL LENGTH
plt.figure(figsize=(10, 5))

plt.plot(df.index[:100], df['bill_length_mm'][:100],
         marker='o', markersize=3)

plt.title("Bill Length of First 100 Penguins")
plt.xlabel("Record Number")
plt.ylabel("Bill Length (mm)")
plt.grid()
plt.show()


#EXPLORE THE DATASET
print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nSpecies Count:")
print(df['species'].value_counts())


# VISUALIZATION 2: SPECIES DISTRIBUTION
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='species')

plt.title("Distribution of Penguin Species")
plt.xlabel("Penguin Species")
plt.ylabel("Number of Penguins")
plt.show()


#SELECT FEATURES AND TARGET
X = df[['bill_length_mm',
        'bill_depth_mm',
        'flipper_length_mm',
        'body_mass_g']]

y = df['species']

print("\nSelected Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

print("\nMissing Values:")
print(X.isnull().sum())


# VISUALIZATION 3: FEATURE DISTRIBUTION
X.hist(figsize=(12, 8), bins=20, edgecolor='black')

plt.suptitle("Distribution of Penguin Features")
plt.tight_layout()
plt.show()

# VISUALIZATION 4: SPECIES RELATIONSHIP
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x='bill_length_mm',
    y='bill_depth_mm',
    hue='species',
    s=50
)

plt.title("Bill Length vs Bill Depth")
plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.show()


#CORRELATION AND OUTLIERS
print("\nFeature Correlation:")
print(X.corr())


# VISUALIZATION 5: CORRELATION HEATMAP
plt.figure(figsize=(8, 6))

sns.heatmap(
    X.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title("Feature Correlation Heatmap")
plt.show()


# VISUALIZATION 6: OUTLIER DETECTION
plt.figure(figsize=(11, 6))

sns.boxplot(data=X)

plt.title("Boxplot of Penguin Measurements")
plt.xticks(rotation=20)
plt.show()

#sPLIT DATASET INTO TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# VISUALIZATION 7: TRAIN-TEST SPLIT
plt.figure(figsize=(7, 5))

plt.pie(
    [len(X_train), len(X_test)],
    labels=['Training Data (80%)', 'Testing Data (20%)'],
    autopct='%1.1f%%',
    startangle=90
)

plt.title("Training and Testing Data Distribution")
plt.show()

#FEATURE SCALING
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nOriginal Training Data:")
print(X_train.head())

print("\nScaled Training Data:")
print(pd.DataFrame(
    X_train_scaled,
    columns=X.columns
).head())

# VISUALIZATION 8: BEFORE AND AFTER SCALING
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=X_train, ax=axes[0])

axes[0].set_title("Before Scaling")
axes[0].tick_params(axis='x', rotation=30)

sns.boxplot(
    data=pd.DataFrame(X_train_scaled, columns=X.columns),
    ax=axes[1]
)

axes[1].set_title("After Scaling")
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()


#EUCLIDEAN DISTANCE FUNCTION
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

distance = euclidean_distance(
    X_test_scaled[0],
    X_train_scaled[0]
)
print("\nEuclidean Distance:", round(distance, 4))

# VISUALIZATION 9: FEATURE-WISE DISTANCE
differences = np.abs(
    X_test_scaled[0] - X_train_scaled[0]
)
plt.figure(figsize=(9, 5))

plt.bar(X.columns, differences)

plt.title("Feature-wise Difference Between Two Samples")
plt.xlabel("Features")
plt.ylabel("Absolute Difference")
plt.xticks(rotation=20)
plt.show()


#IMPLEMENT KNN ALGORITHM
def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_sample in X_test:
        distances = []
        for i in range(len(X_train)):
            dist = euclidean_distance(
                test_sample,
                X_train[i]
            )
            distances.append((dist, y_train.iloc[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        labels = [
            label for dist, label in neighbors
        ]
        prediction = Counter(labels).most_common(1)[0][0]
        predictions.append(prediction)
    return np.array(predictions)
print("\nKNN Algorithm Implemented Successfully!")


# VISUALIZATION 10: SEVEN NEAREST NEIGHBORS
distances = []
for i in range(len(X_train_scaled)):
    d = euclidean_distance(
        X_test_scaled[0],
        X_train_scaled[i]
    )

    distances.append((d, y_train.iloc[i]))
distances.sort(key=lambda x: x[0])
nearest = distances[:7]
plt.figure(figsize=(9, 5))
plt.barh(
    [item[1] for item in nearest],
    [item[0] for item in nearest]
)

plt.gca().invert_yaxis()
plt.title("Seven Nearest Neighbors of Test Sample 1")
plt.xlabel("Euclidean Distance")
plt.ylabel("Neighbor Species")
plt.show()


#EXPERIMENT WITH K = 3, 5 AND 7
k_values = [3, 5, 7]
accuracy_results = []
for k in k_values:
    y_pred = knn_predict(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k
    )
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results.append(accuracy * 100)
    print("\nK =", k)
    print("Accuracy:", round(accuracy * 100, 2), "%")

# VISUALIZATION 11: ACCURACY COMPARISON
plt.figure(figsize=(8, 5))
bars = plt.bar(
    ['K = 3', 'K = 5', 'K = 7'],
    accuracy_results,
    edgecolor='black'
)
plt.title("KNN Accuracy Comparison")
plt.xlabel("K Values")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)

for bar, value in zip(bars, accuracy_results):

    plt.text(
        bar.get_x() + bar.get_width()/2,
        value + 1,
        f'{value:.2f}%',
        ha='center'
    )
plt.show()

#MANUAL DISTANCE CALCULATION
for i in range(5):
    print("\n" + "=" * 50)
    print("TEST SAMPLE:", i + 1)
    print("Actual Species:", y_test.iloc[i])
    distances = []
    for j in range(len(X_train_scaled)):
        d = np.sqrt(
            (X_test_scaled[i][0] - X_train_scaled[j][0])**2 +
            (X_test_scaled[i][1] - X_train_scaled[j][1])**2 +
            (X_test_scaled[i][2] - X_train_scaled[j][2])**2 +
            (X_test_scaled[i][3] - X_train_scaled[j][3])**2
        )
        distances.append((d, y_train.iloc[j]))
    distances.sort(key=lambda x: x[0])
    print("\nSeven Nearest Neighbors:")
    for distance, species in distances[:7]:
        print(
            "Distance:", round(distance, 4),
            "| Species:", species
        )
    for k in [3, 5, 7]:
        labels = [
            species for distance, species in distances[:k]
        ]
        prediction = Counter(labels).most_common(1)[0][0]
        print("Prediction for K =", k, ":", prediction)

# VISUALIZATION 12: FIRST FIVE TEST SAMPLES
for i in range(5):
    distances = []
    for j in range(len(X_train_scaled)):
        d = np.sqrt(
            np.sum(
                (X_test_scaled[i] - X_train_scaled[j])**2
            )
        )
        distances.append((d, y_train.iloc[j]))
    distances.sort(key=lambda x: x[0])
    nearest = distances[:7]
    plt.figure(figsize=(8, 4))
    plt.bar(
        range(1, 8),
        [item[0] for item in nearest]
    )
    plt.xticks(
        range(1, 8),
        [item[1] for item in nearest],
        rotation=20
    )
    plt.title("Nearest Neighbors - Test Sample " + str(i+1))
    plt.xlabel("Neighbor Species")
    plt.ylabel("Euclidean Distance")
    plt.show()

#CONFUSION MATRIX
for k in [3, 5, 7]:
    y_pred = knn_predict(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k
    )
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=['Adelie', 'Chinstrap', 'Gentoo']
    )
    print("\nConfusion Matrix for K =", k)
    print(cm)

# VISUALIZATION 13: CONFUSION MATRICES
for k in [3, 5, 7]:
    y_pred = knn_predict(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k
    )
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=['Adelie', 'Chinstrap', 'Gentoo']
    )
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Adelie', 'Chinstrap', 'Gentoo'],
        yticklabels=['Adelie', 'Chinstrap', 'Gentoo']
    )
    plt.title("Confusion Matrix for K = " + str(k))
    plt.xlabel("Predicted Species")
    plt.ylabel("Actual Species")
    plt.show()

#CLASSIFICATION REPORT
for k in [3, 5, 7]:
    y_pred = knn_predict(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k
    )
    print("\nClassification Report for K =", k)
    print(
        classification_report(
            y_test,
            y_pred
        )
    )
    
# VISUALIZATION 14: PRECISION, RECALL AND F1-SCORE
metrics = []
for k in [3, 5, 7]:
    y_pred = knn_predict(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )
    metrics.append([precision, recall, f1])
metrics_df = pd.DataFrame(
    metrics,
    index=['K=3', 'K=5', 'K=7'],
    columns=['Precision', 'Recall', 'F1-score']
)

print("\nEvaluation Metrics:")
print(metrics_df)
metrics_df.plot(
    kind='bar',
    figsize=(9, 5),
    ylim=(0, 1.05)
)
plt.title("KNN Evaluation Metrics")
plt.xlabel("K Values")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend()
plt.show()
