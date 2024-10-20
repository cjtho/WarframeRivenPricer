import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from warframe_marketplace_predictor.filepaths import *
from warframe_marketplace_predictor.training.preprocessors.liquidity_model_preprocessor import Preprocessor, \
    get_model_architecture


def plot_classification_performance(y_test, y_test_pred_proba, threshold=0.5):
    # Convert predicted probabilities to binary predictions based on the threshold
    y_test_pred = (y_test_pred_proba >= threshold).astype(int)

    # Calculate classification performance metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    # Print the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"Confusion Matrix:\n{cm}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def main(show_graph: bool):
    # Read in data
    try:
        df = pd.read_csv(marketplace_dataframe_file_path)
    except FileNotFoundError as f:
        print("Original Error:", f)
        print("You need to run 'auto_setup' first.")
        exit()

    # Quick data examination
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.width", None)  # No max width for display
    pd.set_option("display.max_colwidth", None)  # No limit on column width
    print(df.head())
    print(df.shape)
    print(df.isnull().sum())
    print(df.columns)
    print(df.describe())
    print(f"Class distribution before sampling:\n{df['has_sold'].value_counts()}")

    # Separate the "has_sold" dataframe
    df_sold = df[df["has_sold"] == 1]
    df_not_sold = df[df["has_sold"] == 0]

    # Split the not_sold into bad sellers and can’t tell sellers
    oldest_sold_item_by_percentile = df_sold["days_listed"].quantile(0.90)
    df_sold = df_sold[df_sold["days_listed"] < oldest_sold_item_by_percentile]
    df_bad_sellers = df_not_sold[df_not_sold["days_listed"] > oldest_sold_item_by_percentile]
    # df_undetermined_sellers = df_not_sold[df_not_sold["days_listed"] < oldest_sold_item_by_percentile]

    # See synthetic data shapes
    print("Sold df shape:", df_sold.shape)
    print("Bad sellers df shape:", df_bad_sellers.shape)

    # Combine the relevant data for class balancing
    df_combined = pd.concat([df_bad_sellers, df_sold])

    # Dynamically determine majority and minority classes
    class_counts = df_combined['has_sold'].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    # Separate the majority and minority classes
    df_majority = df_combined[df_combined["has_sold"] == majority_class]
    df_minority = df_combined[df_combined["has_sold"] == minority_class]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # Sample with replacement
                                     n_samples=len(df_majority),  # Match number of majority class
                                     random_state=42)  # Reproducible results

    # Combine upsampled minority class with the majority class
    df_balanced = pd.concat([df_minority_upsampled, df_majority])

    # Shuffle the balanced DataFrame
    df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Class distribution after balancing:\n{df['has_sold'].value_counts()}")

    # Prepare your data
    features = ["weapon_url_name",
                "re_rolled",
                "positive1", "positive2", "positive3", "negative",
                "listing_price"]
    target = "has_sold"
    X = df[features]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing data
    preprocessor = Preprocessor()
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    model = get_model_architecture()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Define early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=10,
                                                      min_delta=0.001,
                                                      restore_best_weights=True)

    # Train the model
    model.fit(
        X_train_preprocessed,
        y_train,
        epochs=100,
        validation_split=0.2,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1
    )

    if show_graph:
        y_test_pred = model.predict(X_test_preprocessed)
        plot_classification_performance(y_test, y_test_pred)

    preprocessor = Preprocessor()
    X_preprocessed = preprocessor.fit_transform(X)

    # Rebuild and recompile the model
    model = get_model_architecture()
    model.compile(optimizer="adam", loss="binary_crossentropy")

    # Instantiate the custom callback with the target loss
    best_epoch = early_stopping.best_epoch

    # Retrain the model on the full dataset
    model.fit(
        X_preprocessed,
        y,
        epochs=best_epoch,
        batch_size=128,
        verbose=1
    )

    # Save model.
    model.save(liquidity_model_model_file_path)
    preprocessor.save(liquidity_model_preprocessor_file_path)


def _save_preprocessor():
    # Read in data
    try:
        df = pd.read_csv(marketplace_dataframe_file_path)
    except FileNotFoundError as f:
        print("Original Error:", f)
        print("You need to run 'auto_setup' first.")
        exit()

    # Prepare your data
    features = ["weapon_url_name",
                "positive1", "positive2", "positive3", "negative",
                "listing_price"]

    X = df[features]
    preprocessor = Preprocessor()
    preprocessor.fit_transform(X)
    preprocessor.save(liquidity_model_preprocessor_file_path)


if __name__ == "__main__":
    # _save_preprocessor()
    main(True)
