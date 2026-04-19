import pandas as pd
import json

input_filepath = 'dataset_sintetico_combinado.json'
output_filepath = 'dataset_preparado.csv'

def process_row(row):

    
    damages = row.get('detected_damages', [])
    damages_text = "detected_damages: " + (", ".join(damages) if isinstance(damages, list) and damages else "none")

    statements = row.get('statements', [])
    formatted_statements = []
    valid_roles = ['insured_driver', 'third_party_driver', 'impartial_witness']

    
    if isinstance(statements, list):
        for stmt in statements:
            role = stmt.get('role', 'unknown_role')
            
            if role not in valid_roles:
                continue 
                
            text = stmt.get('text', '').strip()
            if text: 
                formatted_statements.append(f"{role}: {text}")
                
    statements_text = " | ".join(formatted_statements)
    
    return f"{damages_text} | {statements_text}"

def create_dataset():
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Erro ao carregar {input_filepath}: {e}")
        return
        
    df = pd.DataFrame(data)
    
    required_cols = ['claim_id', 'ground_truth_label', 'statements']
    if not all(col in df.columns for col in required_cols):
        print("Erro: Colunas em falta no JSON.")
        return

    df['X_text'] = df.apply(process_row, axis=1)
    df['Y_label'] = df['ground_truth_label'].apply(
        lambda x: 0 if x == 'genuine_accident' else 1
    )
    
    final_df = df[['claim_id', 'X_text', 'Y_label']]
    final_df.to_csv(output_filepath, index=False, encoding='utf-8')
    print(f"Dataset guardado em '{output_filepath}' com {len(final_df)} casos.")

if __name__ == '__main__':
    create_dataset()
