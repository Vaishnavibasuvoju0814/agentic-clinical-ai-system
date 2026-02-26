from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import numpy as np


class DiseasePredictionAgent:

    def run(self, input_data):

        X = input_data["X"]
        y = input_data["y"]
        disease = input_data["disease"]

        # ---------------- Train / Val / Test Split (70/15/15) ----------------
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # ==========================================================
        # 1️⃣ Logistic Regression
        # ==========================================================
        lr_model = LogisticRegression(max_iter=3000, class_weight="balanced")
        lr_model.fit(X_train, y_train)

        lr_probs = lr_model.predict_proba(X_test)[:, 1]
        lr_preds = (lr_probs >= 0.5).astype(int)

        lr_metrics = {
            "accuracy": accuracy_score(y_test, lr_preds),
            "recall": recall_score(y_test, lr_preds),
            "f1": f1_score(y_test, lr_preds),
            "auc": roc_auc_score(y_test, lr_probs)
        }

        # ==========================================================
        # 2️⃣ Random Forest
        # ==========================================================
        rf_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        )

        rf_model.fit(X_train, y_train)

        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        rf_preds = (rf_probs >= 0.5).astype(int)

        rf_metrics = {
            "accuracy": accuracy_score(y_test, rf_preds),
            "recall": recall_score(y_test, rf_preds),
            "f1": f1_score(y_test, rf_preds),
            "auc": roc_auc_score(y_test, rf_probs)
        }

        # ==========================================================
        # 3️⃣ LightGBM
        # ==========================================================
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        scale_weight = neg / pos if pos != 0 else 1

        lgb_model = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=48,
            scale_pos_weight=scale_weight,
            random_state=42,
            verbose=-1
        )

        lgb_model.fit(X_train, y_train)

        lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
        lgb_preds = (lgb_probs >= 0.5).astype(int)

        lgb_metrics = {
            "accuracy": accuracy_score(y_test, lgb_preds),
            "recall": recall_score(y_test, lgb_preds),
            "f1": f1_score(y_test, lgb_preds),
            "auc": roc_auc_score(y_test, lgb_probs)
        }

        # ==========================================================
        # 🔥 Select Best Model (based on AUC - medically better metric)
        # ==========================================================
        models = {
            "Logistic Regression": (lr_model, lr_metrics, lr_probs),
            "Random Forest": (rf_model, rf_metrics, rf_probs),
            "LightGBM": (lgb_model, lgb_metrics, lgb_probs)
        }

        best_model_name = max(models, key=lambda m: models[m][1]["auc"])
        selected_model, selected_metrics, selected_probs = models[best_model_name]

        return {
            "disease": disease,
            "lr_metrics": lr_metrics,
            "rf_metrics": rf_metrics,
            "lgb_metrics": lgb_metrics,
            "best_model": best_model_name,
            "best_metrics": selected_metrics,
            "model": selected_model,
            "X_test": X_test,
            "y_test": y_test,
            "selected_probs": selected_probs
        }