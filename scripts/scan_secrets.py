import os
import re

secret_patterns = [
    re.compile(r'(?i)(api[_-]?key|access[_-]?token|secret[_-]?key|password|auth[_-]?token)\s*[:=]\s*[\'"][^\'"]+[\'"]'),
    re.compile(r'-----BEGIN (?:RSA|EC|DSA|OPENSSH) PRIVATE KEY-----'),
    re.compile(r'ghp_[a-zA-Z0-9]{36}'),
    re.compile(r'github_pat_[a-zA-Z0-9_]{82}')
]

files_to_check = [
    '.gitignore', 'README.md', 'AI_DISCLOSURE.md', 'predict.py', 'results.csv',
    'predictions.jsonl', 'docker/Dockerfile', 'docker/requirements.txt',
    'model/model_conversion_script.py', 'scripts/generate_predictions_jsonl.py',
    'tests/test_predict_interface.py', 'reports/final_submission_readiness_report.md'
]

flagged = []
for root, dirs, files in os.walk('submission'):
    for f in files:
        files_to_check.append(os.path.join(root, f))

for f in set(files_to_check):
    if os.path.exists(f) and not f.endswith(('.tflite', '.png', '.task', '.npz', '.bin')):
        with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
            content = fp.read()
            for p in secret_patterns:
                m = p.search(content)
                if m:
                    flagged.append((f, m.group(0)))

if flagged:
    print('WARNING: Secrets found:', flagged)
    exit(1)
else:
    print('SECRET SCAN PASS: Zero API keys, private tokens, or credentials found in deliverables.')
