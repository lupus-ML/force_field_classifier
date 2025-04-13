# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# ==============================================================================

# Load Dataset
df = pd.read_csv('particle_dataset.csv')

# Prepare Data
X = df[['x0', 'y0', 'vx0', 'vy0']]
y = df['label'].map({'Stable': 0, 'Chaotic': 1})

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Model
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Stable', 'Chaotic']))

# Confusion Matrix
plt.style.use('dark_background')
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=['Stable', 'Chaotic'])
plt.title("Confusion Matrix", color='white')
plt.grid(False)
plt.show()

# ==============================================================================

# Only use first two features for 2D plotting
feature_names = ['x0', 'y0']
X_vis = df[feature_names]
y_vis = df['label'].map({'Stable': 0, 'Chaotic': 1})

# Train a new tree just on these 2 features for visualization
clf_vis = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_vis.fit(X_vis, y_vis)

# Create a mesh grid
x_min, x_max = X_vis['x0'].min() - 1, X_vis['x0'].max() + 1
y_min, y_max = X_vis['y0'].min() - 1, X_vis['y0'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# Predict over the grid
Z = clf_vis.predict(grid).reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 8))
plt.style.use('dark_background')
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)

# Plot actual data points
scatter = plt.scatter(X_vis['x0'], X_vis['y0'], c=y_vis, cmap='coolwarm', edgecolor='white')
plt.xlabel('x0', color='white')
plt.ylabel('y0', color='white')
plt.title('Decision Boundary: Initial Position → Stability', color='white')
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(*scatter.legend_elements(), title="Class", loc='upper right', facecolor='black', edgecolor='white')
plt.show()