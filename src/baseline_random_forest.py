
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------
# the forest classifier need 'scikit-learn' to be installed in addition to other dependencies. 
# --------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# ------------------
# ourrproject paths.
# ------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

CLASS_NAMES = ["N", "S", "V", "F"]
RANDOM_STATE = 0

# --------------------------------------------------------
# load the preprocessed splits of the dataset into memory.
# --------------------------------------------------------

def load_split(name):
    """Load one preprocessed dataset split."""
    path = DATA_DIR / f"{name}.npz"

    with np.load(path, allow_pickle=True) as data:
        X = data["windows"].astype(np.float32)
        y = data["labels"].astype(str)


    X = X.reshape(X.shape[0], -1)

    return X, y

# --------------------------------------------------------------------
# main function for trainging + evaluation of the random forest model. 
# --------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print("Dataset shapes:")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    print("\nTraining class distribution:")
    for label in CLASS_NAMES:
        print(f"{label}: {(y_train == label).sum()}")

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # --------------------------------
    # Evaluate on the validation data
    # --------------------------------

    val_pred = model.predict(X_val)

    print("\nValidation results:")
    print(f"Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(
        "Macro F1:",
        f"{f1_score(y_val, val_pred, average='macro', zero_division=0):.4f}",
    )

    print("\nValidation classification report:")
    print(
        classification_report(
            y_val,
            val_pred,
            labels=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("Validation confusion matrix:")
    print(confusion_matrix(y_val, val_pred, labels=CLASS_NAMES))

    # --------------------------------
    # final evaluation on test set.
    # --------------------------------

    test_pred = model.predict(X_test)

    print("\nTest results:")
    print(f"Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(
        "Macro F1:",
        f"{f1_score(y_test, test_pred, average='macro', zero_division=0):.4f}",
    )

    print("\nTest classification report:")
    print(
        classification_report(
            y_test,
            test_pred,
            labels=CLASS_NAMES,
            zero_division=0,
        )
    )

    # ------------------------------
    # confusion matrix for test set.
    # ------------------------------

    cm = confusion_matrix(y_test, test_pred, labels=CLASS_NAMES)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )
    display.plot(values_format="d")
    plt.title("Random Forest Baseline - Test Set")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "random_forest_confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    # ----------------------------------------
    # save the metrics for a later comparison.
    # ----------------------------------------

    metrics_path = RESULTS_DIR / "random_forest_metrics.csv"

    with metrics_path.open("w", encoding="utf-8") as file:
        file.write("split,accuracy,macro_f1\n")
        file.write(
            f"validation,{accuracy_score(y_val, val_pred):.6f},"
            f"{f1_score(y_val, val_pred, average='macro', zero_division=0):.6f}\n"
        )
        file.write(
            f"test,{accuracy_score(y_test, test_pred):.6f},"
            f"{f1_score(y_test, test_pred, average='macro', zero_division=0):.6f}\n"
        )

    # ----------------------------
    # print output file locations.
    # ----------------------------

    print(
        f"\nSaved confusion matrix to: "
        f"{RESULTS_DIR / 'random_forest_confusion_matrix.png'}" #current naming of the confusion-matrix file. Saved as png.
    )
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()