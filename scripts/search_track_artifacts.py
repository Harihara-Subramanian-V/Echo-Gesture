import os

keywords = ['validate.py', 'predictions.jsonl', 'results.csv', 'track 02', 'track 2', 'official manifest', 'test_manifest']
found = []
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in {'.git', '.venv', 'build', '.gradle', '__pycache__'}]
    for f in files:
        if f.endswith(('.md', '.py', '.txt', '.json', '.csv', '.yaml', '.yml')):
            p = os.path.join(root, f)
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read().lower()
                    for kw in keywords:
                        if kw in content:
                            found.append((kw, p))
            except Exception:
                pass

for kw, p in found:
    print(f'Match "{kw}" in {p}')
