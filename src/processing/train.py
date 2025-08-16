import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/gestures.csv")
    ap.add_argument("--out", default="models/gesture_svm.pkl")
    args = ap.parse_args()

    Path("models").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== Report ===")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    dump(clf, args.out)
    print(f"[OK] Model saved to {args.out}")

if __name__ == "__main__":
    main()
