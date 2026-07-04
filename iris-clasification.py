import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target_names[iris.target], name="Species")

print("=" * 55)
print("        IRIS FLOWER CLASSIFICATION")
print("=" * 55)
print(f"\nDataset shape : {X.shape}")
print(f"Classes       : {y.unique().tolist()}")
print(f"\nSample data:")
print(pd.concat([X, y], axis=1).head())


fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Iris Dataset – Feature Distributions by Species",
             fontsize=15, fontweight="bold")

colors = {"setosa": "#E74C3C", "versicolor": "#2ECC71", "virginica": "#3498DB"}

for ax, feature in zip(axes.flat, X.columns):
    for species in y.unique():
        data = X[feature][y == species]
        ax.hist(data, alpha=0.6, label=species,
                color=colors[species], bins=15, edgecolor="white")
    ax.set_title(feature.replace(" (cm)", ""), fontsize=11)
    ax.set_xlabel("cm")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] eda_distributions.png")

# Pairplot
df_full = pd.concat([X, y], axis=1)
df_full.columns = ["SepalLen", "SepalWid", "PetalLen", "PetalWid", "Species"]

pair_fig = sns.pairplot(df_full, hue="Species",
                        palette=colors, corner=True, plot_kws={"alpha": 0.7})
pair_fig.fig.suptitle("Pairplot of Iris Features", y=1.02, fontsize=13)
pair_fig.savefig("eda_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] eda_pairplot.png")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")


models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":        DecisionTreeClassifier(random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, random_state=42),
}

results = {}
print("\n── Model Accuracy Comparison ──")
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc   = accuracy_score(y_test, preds)
    results[name] = {"model": model, "preds": preds, "accuracy": acc}
    print(f"  {name:<28} {acc * 100:.2f}%")

best_name = max(results, key=lambda k: results[k]["accuracy"])
best      = results[best_name]

print(f"\n✔  Best model: {best_name}  ({best['accuracy']*100:.2f}% accuracy)")
print("\nClassification Report:")
print(classification_report(y_test, best["preds"]))

# Confusion matrix
cm  = confusion_matrix(y_test, best["preds"],
                       labels=["setosa", "versicolor", "virginica"])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["setosa", "versicolor", "virginica"],
            yticklabels=["setosa", "versicolor", "virginica"],
            linewidths=0.5, ax=ax)
ax.set_title(f"Confusion Matrix – {best_name}", fontsize=12, pad=12)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] confusion_matrix.png")

# Accuracy bar chart
fig, ax = plt.subplots(figsize=(8, 4))
names   = list(results.keys())
accs    = [results[n]["accuracy"] * 100 for n in names]
bars    = ax.barh(names, accs, color=["#3498DB", "#2ECC71", "#E74C3C", "#9B59B6"],
                  edgecolor="white", height=0.5)
ax.set_xlim(80, 102)
ax.set_xlabel("Accuracy (%)", fontsize=11)
ax.set_title("Model Comparison – Iris Classification", fontsize=12)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{acc:.2f}%", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] model_comparison.png")

new_samples = pd.DataFrame({
    "sepal length (cm)": [5.1, 6.7, 6.3],
    "sepal width (cm)":  [3.5, 3.0, 2.5],
    "petal length (cm)": [1.4, 5.2, 5.0],
    "petal width (cm)":  [0.2, 2.3, 1.9],
})

new_scaled   = scaler.transform(new_samples)
predictions  = best["model"].predict(new_scaled)

print("\n── Predictions on New Samples ──")
result_df = new_samples.copy()
result_df["Predicted Species"] = predictions
print(result_df.to_string(index=False))

print("\nAll done! ")
