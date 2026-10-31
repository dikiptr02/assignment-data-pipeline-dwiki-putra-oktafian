import os
import numpy as np
import pandas as pd


# ==============================================================================
# 1. TAHAP EXTRACT
# ==============================================================================
def extract(file_path: str) -> pd.DataFrame:
    """Membaca dataset mentah dari file CSV di folder data/raw/."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] File tidak ditemukan di path: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"[EXTRACT] Berhasil membaca {len(df)} baris x {len(df.columns)} kolom dari '{file_path}'")
    return df


# ==============================================================================
# 2. TAHAP INSPECT (PROFILING)
# ==============================================================================
def inspect(df: pd.DataFrame, stage_label: str = "INITIAL") -> dict:
    """Memeriksa kondisi data: dimensi, duplikasi, dan missing values."""
    n_rows, n_cols = df.shape
    n_dups = int(df.duplicated().sum())
    n_missing = int(df.isna().sum().sum())
    missing_by_col = df.isna().sum()[df.isna().sum() > 0].to_dict()

    print(f"\n{'='*20} INSPEKSI KONDISI DATA: {stage_label} {'='*20}")
    print(f"Dimensi Data       : {n_rows} baris x {n_cols} kolom")
    print(f"Jumlah Duplikat    : {n_dups} baris")
    print(f"Total Missing Value: {n_missing} nilai")
    if missing_by_col:
        print("Missing per Kolom  :")
        for col, count in missing_by_col.items():
            print(f"  - {col}: {count} baris ({count / n_rows * 100:.2f}%)")
    else:
        print("Missing per Kolom  : 0 (Data bersih dari missing values)")
    print("=" * 66)

    return {
        "rows": n_rows,
        "columns": n_cols,
        "duplicates": n_dups,
        "missing_total": n_missing,
        "missing_by_col": missing_by_col
    }


# ==============================================================================
# 3. TAHAP DATA CLEANING
# ==============================================================================
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan kolom artifisial, duplikasi, inkonsistensi string, dan missing values."""
    df_clean = df.copy()

    # 1. Drop kolom artifisial non-fitur mobil
    if "transaction_date" in df_clean.columns:
        df_clean = df_clean.drop(columns=["transaction_date"])
        print("[CLEAN] Kolom 'transaction_date' berhasil dihapus.")

    # 2. Hapus duplikasi baris penuh
    dups_count = df_clean.duplicated().sum()
    if dups_count > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"[CLEAN] {dups_count} baris duplikat berhasil dihapus (Sisa: {len(df_clean)} baris).")

    # 3. Standarisasi format string: trim whitespace & lowercase
    str_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        df_clean[col] = df_clean[col].replace(["nan", "none", ""], np.nan)
    print(f"[CLEAN] Format string pada {len(str_cols)} kolom kategorikal telah distandarisasi.")

    # 4. Imputasi Missing Values
    # Numerik: Median (robust terhadap skewness dan outlier)
    numeric_impute_cols = ["price", "horsepower", "stroke"]
    for col in numeric_impute_cols:
        if col in df_clean.columns and df_clean[col].isna().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"[CLEAN] Imputasi median pada '{col}': {median_val}")

    # Kategorikal: Modus (mempertahankan kelas dominan)
    if "num-of-doors" in df_clean.columns and df_clean["num-of-doors"].isna().sum() > 0:
        mode_doors = df_clean["num-of-doors"].mode()[0]
        df_clean["num-of-doors"] = df_clean["num-of-doors"].fillna(mode_doors)
        print(f"[CLEAN] Imputasi modus pada 'num-of-doors': '{mode_doors}'")

    if "make" in df_clean.columns and df_clean["make"].isna().sum() > 0:
        mode_make = df_clean["make"].mode()[0]
        df_clean["make"] = df_clean["make"].fillna(mode_make)
        print(f"[CLEAN] Imputasi modus pada 'make': '{mode_make}'")

    if "horsepower-binned" in df_clean.columns and df_clean["horsepower-binned"].isna().sum() > 0:
        df_clean["horsepower-binned"] = df_clean["horsepower-binned"].fillna("low")
        print("[CLEAN] Imputasi kategori dasar pada 'horsepower-binned': 'low'")

    return df_clean


# ==============================================================================
# 4. TAHAP DATA TRANSFORMATION
# ==============================================================================
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Melakukan feature encoding, normalisasi Min-Max, dan penataan kolom referensi."""
    df_tf = df.copy()

    # 1. Binary Mapping: num-of-doors
    door_map = {"two": 0.0, "four": 1.0}
    df_tf["num-of-doors"] = df_tf["num-of-doors"].map(door_map).astype(float)

    # 2. Ordinal to Scaled Range: num-of-cylinders
    cyl_map = {
        "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "eight": 8, "twelve": 12
    }
    df_tf["num-of-cylinders"] = (df_tf["num-of-cylinders"].map(cyl_map) - 2) / 10.0

    # 3. Ordinal Integer Encoding: horsepower_ordinal
    hp_bin_map = {"low": 0, "medium": 1, "high": 2}
    df_tf["horsepower_ordinal"] = (
        df_tf["horsepower-binned"].map(hp_bin_map).fillna(0).astype(int)
    )

    # 4. Frequency Encoding: make_freq
    make_counts = df_tf["make"].value_counts()
    df_tf["make_freq"] = df_tf["make"].map(make_counts) / len(df_tf)

    # 5. Min-Max Scaling [0.0, 1.0] (8 fitur numerik kontinu)
    scale_cols = [
        "symboling", "curb-weight", "engine-size", "horsepower",
        "peak-rpm", "city-mpg", "highway-mpg", "price"
    ]
    for col in scale_cols:
        col_min = df_tf[col].min()
        col_max = df_tf[col].max()
        df_tf[col] = (df_tf[col] - col_min) / (col_max - col_min)

    # 6. One-Hot Dummy Encoding (6 fitur kategorikal nominal)
    cat_ohe_cols = [
        "body-style", "drive-wheels", "aspiration",
        "engine-type", "engine-location", "fuel-system"
    ]
    for col in cat_ohe_cols:
        dummies = pd.get_dummies(df_tf[col], prefix=col, dtype=int)
        df_tf = pd.concat([df_tf, dummies], axis=1)

    # Drop kolom asli yang telah di-encode
    df_tf = df_tf.drop(columns=cat_ohe_cols + ["make", "horsepower-binned"])

    # 7. Penyelarasan susunan 50 kolom sesuai standar referensi
    target_columns = [
        "symboling", "normalized-losses", "num-of-doors", "wheel-base", "length",
        "width", "height", "curb-weight", "num-of-cylinders", "engine-size",
        "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
        "city-mpg", "highway-mpg", "price", "city-L/100km", "diesel", "gas",
        "horsepower_ordinal", "body-style_convertible", "body-style_hardtop",
        "body-style_hatchback", "body-style_sedan", "body-style_wagon",
        "drive-wheels_4wd", "drive-wheels_awd", "drive-wheels_fwd", "drive-wheels_rwd",
        "aspiration_std", "aspiration_turbo", "engine-type_dohc", "engine-type_l",
        "engine-type_ohc", "engine-type_ohcf", "engine-type_ohcv", "engine-type_rotor",
        "engine-location_front", "engine-location_rear", "fuel-system_1bbl",
        "fuel-system_2bbl", "fuel-system_4bbl", "fuel-system_idi", "fuel-system_mfi",
        "fuel-system_mpfi", "fuel-system_spdi", "fuel-system_spfi", "make_freq"
    ]

    df_tf = df_tf[target_columns]
    print(f"[TRANSFORM] Berhasil mentransformasi data menjadi {df_tf.shape[0]} baris x {df_tf.shape[1]} kolom.")
    return df_tf


# ==============================================================================
# 5. TAHAP LOAD
# ==============================================================================
def load(df: pd.DataFrame, output_path: str) -> None:
    """Menyimpan hasil processed dataset ke folder data/processed/."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"[LOAD] Berhasil menyimpan dataset ke '{output_path}' ({file_size_kb:.2f} KB)")


# ==============================================================================
# 6. PIPELINE ORCHESTRATOR (ETL RUNNER)
# ==============================================================================
def run_pipeline(
    raw_path: str = "data/raw/automobileEDA_dirty_training.csv",
    processed_path: str = "data/processed/automobileEDA_processed.csv"
) -> pd.DataFrame:
    """Menjalankan seluruh pipeline data ETL secara end-to-end dalam satu kali eksekusi."""
    print("=" * 66)
    print("           MEMULAI DATA ENGINEERING PIPELINE (ETL)             ")
    print("=" * 66)

    # 1. EXTRACT
    df_raw = extract(raw_path)

    # 2. INSPECT (Kondisi Awal / Before)
    metrics_before = inspect(df_raw, stage_label="SEBELUM CLEANING (RAW)")

    # 3. TRANSFORM (Clean + Transform)
    df_clean = clean(df_raw)
    df_processed = transform(df_clean)

    # INSPECT (Kondisi Akhir / After)
    metrics_after = inspect(df_processed, stage_label="SESUDAH TRANSFORMASI (PROCESSED)")

    # 4. LOAD
    load(df_processed, processed_path)

    print("\n" + "=" * 66)
    print("              PIPELINE ETL SELESAI DENGAN SUKSES!               ")
    print("=" * 66)

    return df_processed


if __name__ == "__main__":
    # Path relatif terhadap root project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, "data", "raw", "automobileEDA_dirty_training.csv")
    processed_csv = os.path.join(base_dir, "data", "processed", "automobileEDA_processed.csv")

    run_pipeline(raw_path=raw_csv, processed_path=processed_csv)