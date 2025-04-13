# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


# Load Dataset-
df = pd.read_csv('particle_dataset.csv')

print("\nLabel Distribution:")
print(df['label'].value_counts())

# Prepare Features, Labels
X = df[['x0', 'y0', 'vx0', 'vy0']]  
y = df['label'].map({'Stable': 0, 'Chaotic': 1})  

# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Random Forest
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate Model
y_pred = clf.predict(X_test)

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred, target_names=['Stable', 'Chaotic']))

# Confusion Matrix Plot
plt.style.use('dark_background')
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=['Stable', 'Chaotic'])
plt.title("Confusion Matrix — Random Forest", color='white')
plt.grid(False)
plt.show()