import json
import random
import re

INPUT_FILE = 'dataset_sintetico_gemini.json'
OUTPUT_FILE = 'dataset_sintetico_variado.json'

REPLACEMENTS = {
    r"\bi was driving\b": ["I was driving", "I was heading", "I was travelling", "I was on my way", "I was navigating"],
    r"\bthe other driver\b": ["the other driver", "the second vehicle", "the individual driving the other car", "the oncoming driver"],
    r"\bthere is a\b": ["there is a", "I noticed a", "you can see a", "it left a", "it resulted in a"],
    r"\bscratch on the\b": ["scratch on the", "scrape along the", "mark on the"],
    r"\bdent on the\b": ["dent on the", "damage to the", "impact depression on the", "crush on the"],
    r"\bi didn't see\b": ["I didn't see", "I failed to notice", "I could not spot", "it slipped my attention", "I missed seeing"],
    r"\bi tried to\b": ["I tried to", "I attempted to", "I made an effort to", "I did my best to"],
    r"\bit was a\b": ["it was a", "it happened to be a", "the situation involved a"]
}

def augment_text(text):
    augmented = text
    for pattern, choices in REPLACEMENTS.items():
        if random.random() < 0.6:  # 60% chance to swap matching patterns
            def replace_match(match):
                # Check if it was capitalized to keep consistency if possible (basic version)
                choice = random.choice(choices)
                if match.group(0)[0].isupper() and len(match.group(0)) > 1:
                    if len(choice) > 1 and match.group(0)[1].islower():
                        return choice[0].upper() + choice[1:]
                return choice
            
            augmented = re.sub(pattern, replace_match, augmented, flags=re.IGNORECASE)
    return augmented

def main():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Erro: Ficheiro {INPUT_FILE} não encontrado.")
        return
        
    num_changed = 0
    for item in data:
        changed = False
        if 'statements' in item:
            for stmt in item['statements']:
                if 'text' in stmt:
                    original = stmt['text']
                    new_text = augment_text(original)
                    if new_text != original:
                        stmt['text'] = new_text
                        changed = True
        if changed:
            num_changed += 1

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    print(f"Data augmentation concluded. Saved to {OUTPUT_FILE}.")
    print(f"Modificados {num_changed} testes de {len(data)}.")

if __name__ == "__main__":
    random.seed(42)  # Reprodutibilidade
    main()
