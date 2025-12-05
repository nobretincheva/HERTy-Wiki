import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay


def model_label_to_idx(label):
    answer_options = ["A", "B", "C", "D"]
    if label in answer_options:
        return answer_options.index(label)
    else:
        return np.nan


def handle_sibling_depth(pred_depth, hierarchical_order):
    if pd.notna(pred_depth) and isinstance(hierarchical_order, (list, tuple)):
        return -1 if len(hierarchical_order) == int(pred_depth) + 1 else int(pred_depth)
    return pred_depth


def eval_et(df):
    df["model_idx"] = df["model_label"].apply(model_label_to_idx)

    df["pred_answer"] = df.apply(
        lambda row: row["answer_set"][int(row["model_idx"])]
        if isinstance(row["answer_set"], (list, tuple))
           and pd.notna(row["model_idx"])
           and 0 <= int(row["model_idx"]) < len(row["answer_set"])
        else np.nan,
        axis=1
    )

    df["pred_depth"] = df.apply(
        lambda row: row["hierarchical_order"].index(row["pred_answer"])
        if isinstance(row["hierarchical_order"], (list, tuple))
           and pd.notna(row["pred_answer"])
           and (row["pred_answer"] in row["hierarchical_order"])
        else np.nan,
        axis=1
    )

    df["pred_depth"] = df.apply(
        lambda row: handle_sibling_depth(row["pred_depth"], row["hierarchical_order"])
        if row["depth"] == -1 else row["pred_depth"],
        axis=1
    )

    # =========================
    # METRICS
    # =========================

    # - hierarchical rows: correct if pred_depth equals true depth AND pred_depth is not NaN
    # - sibling rows:      correct if pred_depth == -1; NaN counts as incorrect
    df["correct"] = np.where(
        df["depth"] == -1,
        df["pred_depth"] == -1,
        (df["pred_depth"].notna()) & (df["pred_depth"] == df["depth"])
    )

    hier_mask = df["depth"] != -1
    sib_mask = df["depth"] == -1

    # ---- Hierarchical confusion matrix (4-class: 0,1,2,3) ----
    hier_true = df.loc[hier_mask, "depth"]
    hier_pred = df.loc[hier_mask, "pred_depth"]

    valid_hier = hier_true.notna() & hier_pred.notna()
    h_true = hier_true[valid_hier].astype(int)
    h_pred = hier_pred[valid_hier].astype(int)

    classes_hier = [0, 1, 2, 3]
    cm_hier = confusion_matrix(h_true, h_pred, labels=classes_hier)
    disp_h = ConfusionMatrixDisplay(confusion_matrix=cm_hier,
                                    display_labels=["most specific", "specific", "least specific", "none"])
    disp_h.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix — Hierarchical depths")
    plt.tight_layout()

    # === Save ===
    plt.savefig("confusion_matrix_hierarchical.png", dpi=300)
    plt.close()

    # ---- Hierarchical metrics ----
    acc_hier = accuracy_score(h_true, h_pred) if len(h_true) else np.nan

    if len(h_true):
        prec_h, rec_h, f1_h, _ = precision_recall_fscore_support(
            h_true, h_pred, labels=classes_hier, average="macro", zero_division=0
        )
        prec_h_cls, rec_h_cls, f1_h_cls, support_h_cls = precision_recall_fscore_support(
            h_true, h_pred, labels=classes_hier, average=None, zero_division=0
        )
        per_class_df = pd.DataFrame(
            {
                "precision": prec_h_cls,
                "recall": rec_h_cls,
                "f1": f1_h_cls,
                "n": support_h_cls,
            },
            index=classes_hier,
        )
    else:
        prec_h, rec_h, f1_h = np.nan, np.nan, np.nan
        per_class_df = pd.DataFrame(columns=["precision", "recall", "f1", "n"])

    # ---- Sibling Metrics ----
    # Positive = correct (picked last). NaN -> 0 (incorrect).
    sib_pred_depth = df.loc[sib_mask, "pred_depth"]
    s_y_true = np.ones(len(sib_pred_depth), dtype=int)  # all gold are "correct" class logically
    s_y_pred = sib_pred_depth.eq(-1).fillna(False).astype(int)  # 1 if pred == -1 else 0

    acc_sib = accuracy_score(s_y_true, s_y_pred) if len(s_y_true) else np.nan
    prec_s, rec_s, f1_s, _ = precision_recall_fscore_support(
        s_y_true, s_y_pred, average="binary", pos_label=1, zero_division=0
    ) if len(s_y_true) else (np.nan, np.nan, np.nan, None)

    # -------- Per-depth accuracy (includes sibling) --------
    per_depth = (
        df.loc[hier_mask]
        .groupby("depth", dropna=False)["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
    )

    if not per_class_df.empty:
        per_depth = per_depth.join(per_class_df[["precision", "recall", "f1"]], how="left")

    if sib_mask.any():
        per_depth.loc["sibling", "accuracy"] = df.loc[sib_mask, "correct"].mean()
        per_depth.loc["sibling", "n"] = int(sib_mask.sum())

    # -------- Per-domain macro F1 --------
    f1_by_domain = None
    if "domain" in df.columns:

        # Hierarchy goes from 0 - most specific, to 3 - none correct
        classes_hier = [0, 1, 2, 3]

        domain_results = {}
        for dom, subdf in df.groupby("domain"):

            # restrict to hierarchical rows (excl. sibling case)
            mask = subdf["depth"].notna() & (subdf["depth"] != -1)
            h_true = subdf.loc[mask, "depth"]
            h_pred = subdf.loc[mask, "pred_depth"]

            # sklearn: drop NaNs
            valid = h_true.notna() & h_pred.notna()
            h_true_valid = h_true[valid].astype(int)
            h_pred_valid = h_pred[valid].astype(int)

            # compute macro F1 for this domain
            if len(h_true_valid):
                _, _, f1_macro_dom, _ = precision_recall_fscore_support(
                    h_true_valid,
                    h_pred_valid,
                    labels=classes_hier,
                    average="macro",
                    zero_division=0
                )
            else:
                f1_macro_dom = np.nan

            domain_results[dom] = f1_macro_dom

        f1_by_domain = pd.Series(domain_results, name="macro_f1")

    # -------- Summary results --------
    results = {
        "hierarchical": {
            "accuracy": float(acc_hier) if not np.isnan(acc_hier) else np.nan,
            "precision_macro": float(prec_h) if not (isinstance(prec_h, float) and np.isnan(prec_h)) else np.nan,
            "recall_macro": float(rec_h) if not (isinstance(rec_h, float) and np.isnan(rec_h)) else np.nan,
            "f1_macro": float(f1_h) if not (isinstance(f1_h, float) and np.isnan(f1_h)) else np.nan,
        },
        "sibling": {
            "accuracy": float(acc_sib) if not np.isnan(acc_sib) else np.nan,
            "precision": float(prec_s) if not (isinstance(prec_s, float) and np.isnan(prec_s)) else np.nan,
            "recall": float(rec_s) if not (isinstance(rec_s, float) and np.isnan(rec_s)) else np.nan,
            "f1": float(f1_s) if not (isinstance(f1_s, float) and np.isnan(f1_s)) else np.nan,
        },
        "per_depth_accuracy": per_depth,  # depth 0–3 + sibling
        "macro_f1_by_domain": f1_by_domain,
        "std_per_domain": float(np.std(f1_by_domain.values)),
        "scored_df": df,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Baseline Models")

    parser.add_argument(
        "-i", "--input_file",
        type=str,
        required=True,
        help="Path to the input file"
    )

    parser.add_argument(
        "-r", "--results_file",
        type=str,
        required=True,
        help="Path to the configuration file"
    )

    parser.add_argument("-t", "--task",
                        type=str,
                        required=True,
                        help="Task performance being evaluated: [et]")

    args = parser.parse_args()

    # File paths
    input_file = args.input_file
    results_file = args.results_file

    # Load the input file
    logger.info(f"Loading the input file `{input_file}`...")
    input_rows = pd.read_parquet(input_file)

    # Load the results file
    logger.info(f"Loading the results file `{results_file}`...")
    results_rows = pd.read_json(results_file, orient="records", lines=True)
    if "original_idx" in results_rows.columns:
        results_rows = results_rows.sort_values("original_idx", ascending=True).reset_index(drop=True)

    assert len(input_rows) == len(results_rows)
    out = pd.concat(
        [input_rows.reset_index(drop=True), results_rows.reset_index(drop=True)],
        axis=1
    )

    for col in ["hierarchical_order", "answer_set"]:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: x.tolist() if isinstance(x, (np.ndarray, pd.Series)) else x
            )

    results = eval_et(out)
    print(results)

    logger.info("Done!")


if __name__ == "__main__":
    main()
