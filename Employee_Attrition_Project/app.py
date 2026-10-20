from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pandas as pd
import numpy as np
import io

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = "change-this-key-before-deployment"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

prediction_file = None


def read_employee_csv(file):
    # Support common CSV encodings.
    raw = file.read()

    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                encoding=encoding,
                sep=None,
                engine="python"
            )
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    raise ValueError("Unable to read the CSV file.")


def find_target(columns):
    for column in columns:
        if column.strip().lower() in [
            "attrition", "employee_attrition",
            "attrition_flag", "left"
        ]:
            return column
    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global prediction_file

    file = request.files.get("file")

    if not file or not file.filename:
        flash("Please select a CSV file.", "error")
        return redirect(url_for("home"))

    if not file.filename.lower().endswith(".csv"):
        flash("Only CSV files are accepted.", "error")
        return redirect(url_for("home"))

    try:
        df = read_employee_csv(file)
        df.columns = df.columns.astype(str).str.strip()

        if df.empty or len(df.columns) < 2:
            raise ValueError("Your CSV needs employee rows and columns.")

        target = find_target(df.columns)

        if target is None:
            raise ValueError(
                "Missing Attrition column. Include known Yes/No "
                "attrition outcomes in your training CSV."
            )

        df = df.dropna(subset=[target]).copy()

        # Standardize common binary target values.
        target_values = df[target].astype(str).str.strip()
        mapping = {
            "yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes",
            "no": "No", "n": "No", "false": "No", "0": "No"
        }
        df[target] = target_values.map(
            lambda value: mapping.get(value.lower(), value)
        )

        if df[target].nunique() < 2:
            raise ValueError(
                "Attrition must contain at least two classes, "
                "for example Yes and No."
            )

        X = df.drop(columns=[target]).dropna(axis=1, how="all")
        y = df[target].astype(str)

        if X.shape[1] == 0:
            raise ValueError("No usable employee feature columns found.")

        numeric_cols = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_cols = [
            col for col in X.columns if col not in numeric_cols
        ]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessing = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols)
        ])

        model = Pipeline([
            ("preprocessing", preprocessing),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                class_weight="balanced"
            ))
        ])

        accuracy = None
        class_counts = y.value_counts()

        # Evaluate only when a reasonable stratified split is possible.
        if len(df) >= 10 and class_counts.min() >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            model.fit(X_train, y_train)
            test_predictions = model.predict(X_test)
            accuracy = accuracy_score(
                y_test, test_predictions
            ) * 100

        # Train on all labeled rows for the downloadable output.
        model.fit(X, y)

        output = df.copy()
        output["Predicted_Attrition"] = model.predict(X)

        # Store the download in memory.
        prediction_file = output.to_csv(index=False).encode("utf-8-sig")

        actual_counts = df[target].value_counts().to_dict()
        predicted_counts = (
            output["Predicted_Attrition"].value_counts().to_dict()
        )

        results = {
            "rows": len(df),
            "columns": len(df.columns),
            "target": target,
            "accuracy": round(accuracy, 2) if accuracy is not None else None,
            "actual_counts": actual_counts,
            "predicted_counts": predicted_counts,
            "preview": output.head(10).to_dict(orient="records"),
            "column_names": output.columns.tolist()
        }

        return render_template("index.html", results=results)

    except (ValueError, pd.errors.ParserError) as error:
        flash(str(error), "error")
        return redirect(url_for("home"))

    except Exception:
        app.logger.exception("CSV processing failed")
        flash(
            "The file could not be processed. Check its column names "
            "and data, then try again.",
            "error"
        )
        return redirect(url_for("home"))


@app.route("/download")
def download():
    if prediction_file is None:
        flash("Upload a CSV file before downloading.", "error")
        return redirect(url_for("home"))

    return send_file(
        io.BytesIO(prediction_file),
        mimetype="text/csv",
        as_attachment=True,
        download_name="employee_predictions.csv"
    )


@app.errorhandler(413)
def file_too_large(error):
    flash("The CSV must be smaller than 10 MB.", "error")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
