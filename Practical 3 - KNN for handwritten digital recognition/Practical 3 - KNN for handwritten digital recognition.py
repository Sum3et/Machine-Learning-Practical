#IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

#LOAD DATASET
digits = load_digits()

X = digits.data
y = digits.target

df = pd.DataFrame(X)
df["Digit"] = y

print("FIRST FIVE ROWS:")
print(df.head())

print("\nDATASET SHAPE:", df.shape)
print("NUMBER OF IMAGES:", len(X))
print("IMAGE SIZE: 8 x 8 pixels")

#VISUALIZE SAMPLE DIGITS
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title("Digit: " + str(y[i]))
    ax.axis("off")

plt.suptitle("Sample Handwritten Digits")
plt.tight_layout()
plt.show()

#DISTRIBUTION OF DIGITS
plt.figure(figsize=(8, 4))

plt.hist(y, bins=np.arange(-0.5, 10.5, 1),
         edgecolor="black")

plt.xticks(range(10))
plt.xlabel("Digit")
plt.ylabel("Number of Images")
plt.title("Distribution of Handwritten Digits")
plt.show()

#SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING SAMPLES:", len(X_train))
print("TESTING SAMPLES:", len(X_test))

#FEATURE SCALING
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFEATURE SCALING COMPLETED")

#IMPLEMENT KNN
knn = KNeighborsClassifier(
    n_neighbors=3,
    metric="euclidean"
)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nKNN CLASSIFICATION")
print("K VALUE: 3")
print("DISTANCE METRIC: EUCLIDEAN")
print("ACCURACY:", round(accuracy * 100, 2), "%")

#COMPARE DIFFERENT K VALUES
k_values = [3, 5, 7]
accuracies = []

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = accuracy_score(y_test, predictions)

    accuracies.append(score)

    print("K =", k, "Accuracy =", round(score * 100, 2), "%")

#Visualization
plt.figure(figsize=(7, 4))

plt.bar(
    [str(k) for k in k_values],
    [a * 100 for a in accuracies]
)

plt.xlabel("K Value")
plt.ylabel("Accuracy (%)")
plt.title("KNN Accuracy for Different K Values")
plt.ylim(80, 100)
plt.show()

#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

plt.imshow(cm, cmap="Blues")
plt.colorbar()

plt.xticks(range(10))
plt.yticks(range(10))

plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.title("KNN Confusion Matrix")

for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.show()

#VISUALIZE 3 TEST IMAGES
model = KNeighborsClassifier(
    n_neighbors=3,
    metric="euclidean"
)

model.fit(X_train, y_train)

#Select first three test images
for i in range(3):

    distances, indices = model.kneighbors(
        [X_test[i]],
        n_neighbors=3
    )

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))

#Original test image
    test_image = scaler.inverse_transform(
        [X_test[i]]
    )[0].reshape(8, 8)

    axes[0].imshow(test_image, cmap="gray")
    axes[0].set_title(
        "Test Image\nActual: " + str(y_test[i])
    )
    axes[0].axis("off")

#Three nearest neighbors
    for j in range(3):

        neighbor_index = indices[0][j]

        neighbor_image = scaler.inverse_transform(
            [X_train[neighbor_index]]
        )[0].reshape(8, 8)

        axes[j + 1].imshow(
            neighbor_image, cmap="gray"
        )

        axes[j + 1].set_title(
            "Neighbor " + str(j + 1) +
            "\nDigit: " + str(y_train[neighbor_index]) +
            "\nDist: " + str(round(distances[0][j], 2))
        )

        axes[j + 1].axis("off")

    plt.suptitle(
        "Nearest Neighbors for Test Image " + str(i + 1)
    )

    plt.tight_layout()
    plt.show()


#CLASSIFICATION REPORT
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\nFINAL RESULT:")
print("KNN successfully recognized handwritten digits.")
print("Accuracy:", round(accuracy * 100, 2), "%")
