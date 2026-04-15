import json
import os

def main():
    datasets_files = [
        'dataset_sintetico_variado.json',
        'dataset_sintetico_grok.json',
        'dataset_sintetico_chatgpt.json'
    ]

    combined_data = []

    for filepath in datasets_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                combined_data.extend(data)
                print(f"Lidos {len(data)} casos de {filepath}")
        except Exception as e:
            print(f"Erro ao processar {filepath}: {e}")

    output_filepath = 'dataset_sintetico_combinado.json'
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)

    print(f"Criação do '{output_filepath}' concluída com sucesso!")
    print(f"Total de casos do novo dataset fundido: {len(combined_data)}")

if __name__ == '__main__':
    main()
