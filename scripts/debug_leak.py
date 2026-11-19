import csv

rows = list(csv.DictReader(open(r"c:\EchoCam\data\splits\normalized_multidataset_manifest.csv", encoding="utf-8")))
for r in rows:
    if "Eya_OdUAevw" in r["source_url"]:
        print(f"{r['dataset_source']} | {r['video_id']} | {r['original_label']} | signer:{r['signer_id']} | split:{r['original_split']}")
