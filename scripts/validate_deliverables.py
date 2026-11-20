import os
import json
import csv
import hashlib

def test_jsonl(path):
    print(f"Validating {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"  Total lines: {len(lines)}")
    clip_ids = set()
    for i, line in enumerate(lines):
        obj = json.loads(line)
        for key in ['clip_id', 'prediction', 'confidence', 'top5', 'latency_ms']:
            assert key in obj, f"Missing key {key} in line {i}"
        cid = obj['clip_id']
        assert cid not in clip_ids, f"Duplicate clip_id: {cid}"
        clip_ids.add(cid)
        assert obj['prediction'] == obj['prediction'].upper(), f"Prediction not uppercase: {obj['prediction']}"
        assert isinstance(obj['top5'], list) and len(obj['top5']) == 5, f"Top5 invalid: {obj['top5']}"
        assert 0.0 <= obj['confidence'] <= 1.0, f"Confidence out of range: {obj['confidence']}"
        assert obj['latency_ms'] >= 0.0, f"Latency negative: {obj['latency_ms']}"
    print(f"  PASS: {len(lines)} lines validated, 0 duplicates.")

def test_csv(path):
    print(f"Validating {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
    headers = reader[0]
    expected_headers = ['metric', 'target', 'measured', 'unit', 'device', 'runtime', 'pass']
    assert headers == expected_headers, f"Header mismatch: {headers} vs {expected_headers}"
    rows = reader[1:]
    assert len(rows) == 6, f"Expected 6 rows, got {len(rows)}"
    for r in rows:
        assert len(r) == 7, f"Row length mismatch: {r}"
        print(f"  Row: {r[0]:<25} | target: {r[1]:<20} | measured: {r[2]:<30} | pass: {r[6]}")
    print(f"  PASS: CSV structure and all 6 metrics verified.")

def test_models():
    print("Validating model hashes...")
    paths = [
        'models/quantized/echogesture_exp_d_int8.tflite',
        'android/app/src/main/assets/echogesture_exp_d_int8.tflite',
        'model/echogesture_exp_d_int8.tflite'
    ]
    expected = 'ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499'
    for p in paths:
        assert os.path.exists(p), f"Missing model: {p}"
        with open(p, 'rb') as f:
            data = f.read()
            h = hashlib.sha256(data).hexdigest().upper()
            assert h == expected, f"Hash mismatch in {p}: {h}"
            assert len(data) == 135464, f"Size mismatch in {p}: {len(data)}"
        print(f"  PASS: {p} (135,464 bytes, SHA-256 verified)")

if __name__ == '__main__':
    test_jsonl('submission/predictions.jsonl')
    test_jsonl('predictions.jsonl')
    test_csv('submission/results.csv')
    test_csv('results.csv')
    test_models()
    print("\nALL DELIVERABLE CHECKS PASSED SUCCESSFULLY!")
