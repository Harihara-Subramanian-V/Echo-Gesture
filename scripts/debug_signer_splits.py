"""
Inspect which signer IDs appear across multiple unified splits.
"""

import csv
from collections import defaultdict

rows = list(csv.DictReader(open(r"c:\EchoCam\data\splits\normalized_multidataset_manifest.csv", encoding="utf-8")))

signer_splits = defaultdict(set)
signer_sources = defaultdict(set)
signer_instances = defaultdict(list)

for r in rows:
    sig = r["signer_id"]
    sp = r["unified_split"]
    signer_splits[sig].add(sp)
    signer_sources[sig].add(r["dataset_source"])
    signer_instances[sig].append(r)

multi_split_signers = {s: sps for s, sps in signer_splits.items() if len(sps) > 1}
print(f"Signer IDs spanning multiple unified splits: {len(multi_split_signers)}")
for s, sps in multi_split_signers.items():
    srcs = signer_sources[s]
    count = len(signer_instances[s])
    print(f"  Signer {s} ({srcs}, {count} clips) spans splits: {sps}")
