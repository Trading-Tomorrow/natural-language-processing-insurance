import pandas as pd
import json

input_filepath = '../dataset_insurance.json'
output_filepath = 'dataset_preparado.csv'

def process_statements(statements):
    """Concatena o 'role' e o 'text' num único formato textual."""
    if not statements or not isinstance(statements, list):
        return ""
    
    formatted_statements = []
    for stmt in statements:
        role = stmt.get('role', 'unknown_role')
        text = stmt.get('text', '').strip()
        if text: 
            formatted_statements.append(f"{role}: {text}")
            
    return " | ".join(formatted_statements)

def create_dataset():
    # Carregar os dados
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Erro ao carregar {input_filepath}: {e}")
        return
        
    df = pd.DataFrame(data)
    
    # Validar colunas exigidas
    required_cols = ['claim_id', 'ground_truth_label', 'statements']
    if not all(col in df.columns for col in required_cols):
        print("Erro: Colunas em falta no JSON.")
        return

    # Criar features e labels binárias (0 = genuíno, 1 = fraude)
    df['X_text'] = df['statements'].apply(process_statements)
    df['Y_label'] = df['ground_truth_label'].apply(
        lambda x: 0 if x == 'genuine_accident' else 1
    )
    
    # Exportar DataFrame processado para CSV
    final_df = df[['claim_id', 'X_text', 'Y_label']]
    final_df.to_csv(output_filepath, index=False, encoding='utf-8')
    print(f"Dataset guardado em '{output_filepath}' com {len(final_df)} casos.")

if __name__ == '__main__':
    create_dataset()
