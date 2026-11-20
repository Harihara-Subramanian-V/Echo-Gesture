import urllib.request
import json

urls = [
    'https://raw.githubusercontent.com/Harihara-Subramanian-V/Echo-Gesture/main/README.md',
    'https://raw.githubusercontent.com/Harihara-Subramanian-V/Echo-Gesture/main/results.csv',
    'https://raw.githubusercontent.com/Harihara-Subramanian-V/Echo-Gesture/main/AI_DISCLOSURE.md',
    'https://raw.githubusercontent.com/Harihara-Subramanian-V/Echo-Gesture/main/predict.py',
    'https://raw.githubusercontent.com/Harihara-Subramanian-V/Echo-Gesture/main/submission/results.csv',
    'https://raw.githubusercontent.com/Harihara-Subramanian-V/Echo-Gesture/main/submission/predictions.jsonl'
]

print("Verifying GitHub API Repository Metadata...")
req = urllib.request.Request('https://api.github.com/repos/Harihara-Subramanian-V/Echo-Gesture', headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        repo_data = json.loads(response.read().decode())
        print(f"  Repository: {repo_data.get('full_name')}")
        print(f"  Default Branch: {repo_data.get('default_branch')}")
        print(f"  Is Private: {repo_data.get('private')}")
        print(f"  HTML URL: {repo_data.get('html_url')}")
        print(f"  Public Access: {'CONFIRMED (Public)' if not repo_data.get('private') else 'PRIVATE'}")
except Exception as e:
    print(f"  GitHub API Query: {e}")

print("\nVerifying Raw Content Direct URLs...")
for u in urls:
    try:
        req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            name = u.replace('https://raw.githubusercontent.com/Harihara-Subramanian-V/Echo-Gesture/main/', '')
            print(f"  PASS [{resp.status}]: {name} ({len(content)} bytes)")
    except Exception as e:
        print(f"  FAIL: {u} - {e}")
