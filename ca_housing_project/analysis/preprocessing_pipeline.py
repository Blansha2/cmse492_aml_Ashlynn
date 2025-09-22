from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

#I used chatgpt to help with creating the data pipline.

def project_paths(): #etting up the file paths
    proj = Path(__file__).resolve().parents[1]
    data_train = proj / "data" / "train"
    return proj, data_train

def build_preprocessor(numeric_cols, categorical_cols): #building the preprocessor
    num_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),)

    try:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        cat_encoder,)

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0, )
    return pre

def main():
    proj, data_train = project_paths()
    in_path = data_train / "housing_train.csv"
    out_path = data_train / "housing_train_processed.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Could not find training CSV at: {in_path}")

    df = pd.read_csv(in_path)

    target_col = "median_house_value"
    if target_col not in df.columns:
        raise KeyError(f"Expected target column '{target_col}' in {in_path}")
    y = df[target_col].copy()

    cat_cols = ["ocean_proximity"]
    num_cols = [c for c in df.columns if c not in cat_cols + [target_col]]

    pre = build_preprocessor(num_cols, cat_cols)

    X = df[num_cols + cat_cols]
    X_proc = pre.fit_transform(X)
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        
        feat_names = [f"f{i}" for i in range(X_proc.shape[1])]

    feat_names = [n.replace("num__", "").replace("cat__", "") for n in feat_names]

    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    df_proc = pd.DataFrame(X_proc, columns=feat_names, index=df.index)

    df_proc[target_col] = y.values

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_proc.to_csv(out_path, index=False)
    print(f"Saved processed training set: {out_path.resolve()}  shape={df_proc.shape}")


if __name__ == "__main__":
    main()