# agents/prediction_agent.py

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb


class DiseasePredictionAgent:

    def run(self, input_data):

        X = input_data["X"]
        y = input_data["y"]
        disease = input_data["disease"]

        # 70/15/15 split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # ---------------- Logistic Regression ----------------
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

        # ---------------- LightGBM ----------------
        pos = sum(y_train == 1)
        neg = sum(y_train == 0)
        scale_weight = neg / pos

        lgb_model = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=48,
            scale_pos_weight=scale_weight,
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

        # ---------------- Select Best ----------------
        if lgb_metrics["accuracy"] >= lr_metrics["accuracy"]:
            best_model = "LightGBM"
            selected_model = lgb_model
            selected_probs = lgb_probs
        else:
            best_model = "Logistic Regression"
            selected_model = lr_model
            selected_probs = lr_probs

        return {
            "disease": disease,
            "lr_metrics": lr_metrics,
            "lgb_metrics": lgb_metrics,
            "best_model": best_model,
            "model": selected_model,
            "X_test": X_test,
            "y_test": y_test,
            "selected_probs": selected_probs
        }
