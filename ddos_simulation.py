import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns

# @author hannahgsimon

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
)

# -------------------------
# CLI + REPRODUCIBILITY
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

TIME_STEPS = args.steps
SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# CONFIG
# -------------------------
NUM_SERVERS = 4
SERVER_CAPACITY = 100
ATTACK_START = 20
ATTACK_END = 45


def create_servers():
    return [
        {"name": f"S{i + 1}", "capacity": SERVER_CAPACITY, "health": 100}
        for i in range(NUM_SERVERS)
    ]


def generate_attack(t, strategy):
    attacks = [0] * NUM_SERVERS

    if not (ATTACK_START <= t <= ATTACK_END):
        return attacks

    progress = (t - ATTACK_START) / (ATTACK_END - ATTACK_START)

    if strategy == "targeted":
        attacks[0] = random.randint(150, int(350 * progress + 150))

    elif strategy == "distributed":
        for i in range(NUM_SERVERS):
            attacks[i] = random.randint(50, int(120 * progress + 50))

    elif strategy == "random_burst":
        target = random.randint(0, NUM_SERVERS - 1)
        attacks[target] = random.randint(100, int(320 * progress + 100))

    return attacks


def distribute_traffic(normal_traffic, servers, defense_mode, model=None, prev_loads=None):
    loads = [0] * NUM_SERVERS

    if defense_mode == "none":
        loads[0] = normal_traffic

    elif defense_mode == "static":
        loads = [normal_traffic / NUM_SERVERS] * NUM_SERVERS

    elif defense_mode == "adaptive":
        total_health = sum(server["health"] for server in servers)
        for i, server in enumerate(servers):
            weight = server["health"] / total_health if total_health > 0 else 1 / NUM_SERVERS
            loads[i] = normal_traffic * weight

    elif defense_mode == "ml" and model is not None:
        loads = [normal_traffic / NUM_SERVERS] * NUM_SERVERS

        for i in range(NUM_SERVERS):
            delta = loads[i] - prev_loads[i]

            sample = pd.DataFrame([{
                "load": loads[i],
                "delta_load": delta,
                "capacity": servers[i]["capacity"],
                "health": servers[i]["health"]
            }])

            if model.predict(sample)[0] == 1:
                loads[i] *= 0.5

    return loads


def simulate(defense_mode, attack_strategy, model=None):
    servers = create_servers()

    total_packet_loss, avg_latency, avg_health, total_throughput = [], [], [], []
    data_rows = []
    prev_loads = [0] * NUM_SERVERS

    for t in range(TIME_STEPS):
        normal_traffic = random.randint(120, 180)
        attack_traffic = generate_attack(t, attack_strategy)

        loads = distribute_traffic(normal_traffic, servers, defense_mode, model, prev_loads)

        step_loss = step_latency = step_throughput = 0

        for i, server in enumerate(servers):
            total_load = loads[i] + attack_traffic[i]
            capacity = server["capacity"] * (server["health"] / 100)

            delta = total_load - prev_loads[i]
            is_attack = int(attack_traffic[i] > 0)

            data_rows.append({
                "load": total_load,
                "delta_load": delta,
                "capacity": server["capacity"],
                "health": server["health"],
                "attack": is_attack
            })

            prev_loads[i] = total_load

            if total_load > capacity:
                loss = total_load - capacity
                latency = 1 + (loss / server["capacity"])
                throughput = capacity
                server["health"] -= 8
            else:
                loss = 0
                latency = 1 + random.uniform(0, 0.2)
                throughput = total_load
                server["health"] += 3

            server["health"] = max(0, min(100, server["health"]))

            step_loss += loss
            step_latency += latency
            step_throughput += throughput

        total_packet_loss.append(step_loss)
        avg_latency.append(step_latency / NUM_SERVERS)
        avg_health.append(sum(s["health"] for s in servers) / NUM_SERVERS)
        total_throughput.append(step_throughput)

    return total_packet_loss, avg_latency, avg_health, total_throughput, pd.DataFrame(data_rows)


# -------------------------
# DATA GENERATION
# -------------------------
attack_strategies = ["targeted", "distributed", "random_burst"]
base_modes = ["none", "static", "adaptive"]

dataset = pd.concat(
    [simulate(m, s)[4] for s in attack_strategies for m in base_modes],
    ignore_index=True
)

dataset.to_csv("generated_dataset.csv", index=False)

# -------------------------
# TRAIN / TEST SPLIT
# -------------------------
X = dataset[["load", "delta_load", "capacity", "health"]]
y = dataset["attack"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y  # stratify preserves class balance
)

# -------------------------
# MODELS
# -------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=SEED)

# Logistic Regression needs scaled features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_model = LogisticRegression(random_state=SEED, max_iter=1000)

# Dummy classifier: always predicts the majority class (weakest possible baseline)
dummy_model = DummyClassifier(strategy="most_frequent", random_state=SEED)

# -------------------------
# CROSS-VALIDATION (5-fold stratified)
# -------------------------
print("\n===== CROSS-VALIDATION (5-fold) =====")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for name, model, Xdata in [
    ("Random Forest",      rf_model,    X),
    ("Logistic Regression", lr_model,   scaler.fit_transform(X)),
    ("Dummy (majority)",   dummy_model, X),
]:
    scores = cross_val_score(model, Xdata, y, cv=cv, scoring="f1_weighted")
    print(f"  {name:<25} F1 = {scores.mean():.3f} ± {scores.std():.3f}")

# -------------------------
# FIT FINAL MODELS ON TRAIN SET
# -------------------------
rf_model.fit(X_train, y_train)
lr_model.fit(X_train_scaled, y_train)
dummy_model.fit(X_train, y_train)

# -------------------------
# HELD-OUT TEST EVALUATION
# -------------------------
print("\n===== ML RESULTS (Random Forest) =====")
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))

print("\n===== BASELINE: Logistic Regression =====")
y_pred_lr = lr_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_lr))

print("\n===== BASELINE: Dummy Classifier =====")
y_pred_dummy = dummy_model.predict(X_test)
print(classification_report(y_test, y_pred_dummy, zero_division=0))

# -------------------------
# ABLATION: remove delta_load
# -------------------------
print("\n===== ABLATION =====")
X_wo = dataset[["load", "capacity", "health"]]
Xtr, Xte, ytr, yte = train_test_split(X_wo, y, test_size=0.2, random_state=SEED, stratify=y)
model_wo = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(Xtr, ytr)
print("\nWithout delta_load:")
print(classification_report(yte, model_wo.predict(Xte)))

# -------------------------
# CONFUSION MATRIX
# -------------------------
fig, ax = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf)).plot(ax=ax)
plt.title("Confusion Matrix (Random Forest)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

# -------------------------
# ROC CURVE — all models compared
# -------------------------
plt.figure()
y_probs_rf = rf_model.predict_proba(X_test)[:, 1]
y_probs_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

for label, probs in [("Random Forest", y_probs_rf), ("Logistic Regression", y_probs_lr)]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc(fpr, tpr):.2f})")

plt.plot([0, 1], [0, 1], '--', label="Random", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Model Comparison")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("roc_curve.png")

# -------------------------
# PRECISION-RECALL CURVE — all models compared
# -------------------------
plt.figure()
for label, probs in [("Random Forest", y_probs_rf), ("Logistic Regression", y_probs_lr)]:
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.plot(recall, precision, label=label)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Model Comparison")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("pr_curve.png")

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
plt.figure()
importance = rf_model.feature_importances_
features = X.columns
plt.barh(features, importance)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png")

# -------------------------
# BASELINE COMPARISON BAR CHART
# -------------------------
from sklearn.metrics import f1_score, precision_score, recall_score

comparison_data = {
    "Model": ["Dummy (majority)", "Logistic Regression", "Random Forest"],
    "Precision": [
        precision_score(y_test, y_pred_dummy, average="weighted", zero_division=0),
        precision_score(y_test, y_pred_lr, average="weighted"),
        precision_score(y_test, y_pred_rf, average="weighted"),
    ],
    "Recall": [
        recall_score(y_test, y_pred_dummy, average="weighted", zero_division=0),
        recall_score(y_test, y_pred_lr, average="weighted"),
        recall_score(y_test, y_pred_rf, average="weighted"),
    ],
    "F1": [
        f1_score(y_test, y_pred_dummy, average="weighted", zero_division=0),
        f1_score(y_test, y_pred_lr, average="weighted"),
        f1_score(y_test, y_pred_rf, average="weighted"),
    ],
}
df_comparison = pd.DataFrame(comparison_data)
df_comparison_melted = df_comparison.melt(id_vars="Model", var_name="Metric", value_name="Score")

plt.figure(figsize=(8, 5))
sns.barplot(data=df_comparison_melted, x="Model", y="Score", hue="Metric")
plt.title("Model Comparison: Precision / Recall / F1")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("model_comparison.png")

print("\n===== MODEL COMPARISON =====")
print(df_comparison.to_string(index=False))

# -------------------------
# FINAL SIMULATION (using RF model)
# -------------------------
modes = ["none", "static", "adaptive", "ml"]
all_results = {s: {m: simulate(m, s, rf_model) for m in modes} for s in attack_strategies}

# -------------------------
# SYSTEM GRAPH
# -------------------------
results = all_results["targeted"]

plt.figure(figsize=(10, 8))
titles = ["Packet Loss", "Latency", "Server Health", "Throughput"]
for i in range(4):
    plt.subplot(4, 1, i + 1)
    for mode in modes:
        plt.plot(results[mode][i], label=mode)
    plt.title(titles[i])
    plt.xlabel("Time Step")
    plt.ylabel(titles[i])
    if i == 0:
        plt.legend()

plt.suptitle("System Performance Over Time (Targeted Attack)", fontsize=14)
plt.tight_layout()
plt.savefig("system_performance.png")

# -------------------------
# SUMMARY TABLE
# -------------------------
rows = []
for s in attack_strategies:
    for m in modes:
        rows.append({
            "Strategy": s,
            "Defense": m,
            "Throughput": sum(all_results[s][m][3]) / TIME_STEPS
        })

df_summary = pd.DataFrame(rows)
df_summary.to_csv("results_summary.csv", index=False)

print("\n===== SUMMARY =====\n", df_summary)

# -------------------------
# THROUGHPUT BAR CHART
# -------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=df_summary, x="Strategy", y="Throughput", hue="Defense")
plt.title("Throughput Comparison by Strategy and Defense")
plt.xticks(rotation=30)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("throughput_bar.png")

best = df_summary.sort_values("Throughput", ascending=False).iloc[0]
print(f"\nBest defense: {best['Defense']} under {best['Strategy']} attack")

# -------------------------
# IMPROVEMENT CHART
# -------------------------
df_pivot = df_summary.pivot(index="Strategy", columns="Defense", values="Throughput")
df_pivot["improvement"] = (df_pivot["ml"] - df_pivot["none"]) / df_pivot["none"] * 100

plt.figure()
df_pivot["improvement"].plot(kind="bar")
plt.ylabel("% Improvement")
plt.title("ML Improvement Over No Defense")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("improvement.png")

# -------------------------
# INSIGHTS
# -------------------------
print("\n===== INSIGHTS =====")
ml_throughput = df_summary[
    (df_summary["Strategy"] == "targeted") & (df_summary["Defense"] == "ml")
]["Throughput"].values[0]

none_throughput = df_summary[
    (df_summary["Strategy"] == "targeted") & (df_summary["Defense"] == "none")
]["Throughput"].values[0]

improvement = ((ml_throughput - none_throughput) / none_throughput) * 100
print(f"ML defense improves throughput by {improvement:.1f}% over no defense "
      f"(from {none_throughput:.2f} to {ml_throughput:.2f}) under targeted attack.")

plt.show()