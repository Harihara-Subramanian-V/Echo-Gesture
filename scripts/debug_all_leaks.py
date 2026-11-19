import csv
import re
from collections import defaultdict

def extract_yt_id(url):
    m = re.search(r"(?:v=|\/embed\/|\/watch\?v=|youtu\.be\/|\/v\/)([a-zA-Z0-9_-]{11})", url)
    return m.group(1) if m else None

rows = list(csv.DictReader(open(r"c:\EchoCam\data\splits\normalized_multidataset_manifest.csv", encoding="utf-8")))

yt_splits = defaultdict(set)
yt_rows = defaultdict(list)
for r in rows:
    y = extract_yt_id(r["source_url"])
    if y:
        yt_splits[y].add(r["original_split"])
        yt_rows[y].append(r)

for y, splits in yt_splits.items():
    if len(splits) > 1:
        sources = set(r["dataset_source"] for r in yt_rows[y])
        print(f"YT ID {y} has splits {splits} across sources {sources}:")
        for r in yt_rows[y]:
            print(f"  {r['dataset_source']} | {r['video_id']} | {r['original_label']} | {r['signer_id']} | split:{r['original_split']}")
