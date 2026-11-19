"""
Audit signer independence and splits for MS-ASL and combined dataset.
"""

import json
from collections import defaultdict, Counter

with open(r"c:\EchoCam\data\metadata\ms_asl\msasl_matched_samples.json", "r", encoding="utf-8") as f:
    msasl_samples = json.load(f)

print(f"Total MS-ASL samples: {len(msasl_samples)}")

train_signers = set()
val_signers = set()
test_signers = set()
all_signers = set()

split_counts = Counter()
split_classes = defaultdict(set)

for s in msasl_samples:
    split = s.get("split", "unknown")
    signer = s.get("signer_id")
    gloss = s.get("target_gloss")
    
    split_counts[split] += 1
    split_classes[split].add(gloss)
    all_signers.add(signer)
    
    if split == "train":
        train_signers.add(signer)
    elif split == "val":
        val_signers.add(signer)
    elif split == "test":
        test_signers.add(signer)

print(f"Split instance counts: {dict(split_counts)}")
print(f"Split class coverage: Train={len(split_classes['train'])}/40, Val={len(split_classes['val'])}/40, Test={len(split_classes['test'])}/40")
print(f"Unique signers: Total={len(all_signers)}, Train={len(train_signers)}, Val={len(val_signers)}, Test={len(test_signers)}")

overlap_train_val = train_signers.intersection(val_signers)
overlap_train_test = train_signers.intersection(test_signers)
overlap_val_test = val_signers.intersection(test_signers)

print(f"MS-ASL signer overlap:")
print(f"  Train <-> Val: {len(overlap_train_val)}")
print(f"  Train <-> Test: {len(overlap_train_test)}")
print(f"  Val <-> Test: {len(overlap_val_test)}")

# Also inspect video duration / clip lengths
durations = []
for s in msasl_samples:
    st = s.get("start_time", 0.0)
    et = s.get("end_time", 0.0)
    durations.append(et - st)

avg_dur = sum(durations) / len(durations) if durations else 0
print(f"Average clip duration: {avg_dur:.2f}s (min: {min(durations):.2f}s, max: {max(durations):.2f}s)")
print(f"Estimated storage for 1,994 clips @ ~1.2MB/s: ~{len(msasl_samples) * avg_dur * 0.15:.1f} MB to {len(msasl_samples) * 1.0:.1f} MB")
