import os
import pandas as pd

def update_original_file(df_results, original_path, output_name):
    if not os.path.exists(original_path):
        print(f"File not found: {original_path}")
        return
    df_orig = pd.read_csv(original_path)
    df_orig.columns = df_orig.columns.str.strip()
    
    df_results['filename'] = df_results['filename'].astype(str)
    df_orig['id'] = df_orig['id'].astype(str).str.replace(r'\.(jpg|jpeg|png)$', '', regex=True)
    
    updated = pd.merge(df_orig, df_results[['filename', 'predicted_label']], 
                       left_on='id', right_on='filename', how='left').drop(columns=['filename'])
    
    output_path = os.path.join(os.path.dirname(original_path), output_name)
    updated.to_csv(output_path, index=False)
    print(f"Successfully saved: {output_name}")