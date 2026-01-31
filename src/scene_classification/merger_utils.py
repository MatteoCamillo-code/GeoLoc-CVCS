import os
import pandas as pd

def update_original_file(df_united, original_path, output_name):
    if not os.path.exists(original_path):
        print(f"File not found: {original_path}")
        return

    df_orig = pd.read_csv(original_path)
    df_orig.columns = df_orig.columns.str.strip()
    df_united.columns = df_united.columns.str.strip()

    ORIG_COL = 'id'
    UNITED_COL = 'filename'

    if ORIG_COL not in df_orig.columns:
        print(f"Error: '{ORIG_COL}' not found in {original_path}.")
        return

    df_united[UNITED_COL] = df_united[UNITED_COL].astype(str)
    df_orig[ORIG_COL] = df_orig[ORIG_COL].astype(str).str.replace(r'\.(jpg|jpeg|png)$', '', regex=True)

    updated_df = pd.merge(
        df_orig,
        df_united[[UNITED_COL, 'predicted_label']],
        left_on=ORIG_COL,
        right_on=UNITED_COL,
        how='left'
    )

    if UNITED_COL in updated_df.columns and UNITED_COL != ORIG_COL:
        updated_df = updated_df.drop(columns=[UNITED_COL])

    output_path = os.path.join(os.path.dirname(original_path), output_name)
    updated_df.to_csv(output_path, index=False)
    print(f"Successfully created: {output_name}")