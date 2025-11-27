# Save as revert_to_4omini.py
from pathlib import Path

def revert_model():
    agents_path = Path("agents")
    count = 0
    
    for py_file in agents_path.rglob("*.py"):
        content = py_file.read_text(encoding='utf-8')
        if 'gpt-5-nano' in content:
            new_content = content.replace('gpt-5-nano', 'gpt-4o-mini')
            py_file.write_text(new_content, encoding='utf-8')
            matches = content.count('gpt-5-nano')
            count += matches
            print(f"✓ {py_file.name}: {matches} replacement(s)")
    
    print(f"\nReverted {count} total occurrences to gpt-4o-mini")

if __name__ == "__main__":
    revert_model()