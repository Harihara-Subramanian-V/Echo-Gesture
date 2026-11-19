import pandas as pd
import numpy as np

df_before = pd.read_csv("reports/phase6b_multiclip_baseline_before.csv")
df_after = pd.read_csv("reports/phase6b_multiclip_evaluation.csv")

print("=== PHASE 6B MULTI-CLIP AUDIT: BEFORE vs AFTER ===")
print(f"Total clips evaluated: {len(df_before)}")
print(f"Missing clips (Before): {df_before['is_missing'].sum()} ({df_before['is_missing'].mean()*100:.1f}%)")
print(f"Missing clips (After):  {df_after['is_missing'].sum()} ({df_after['is_missing'].mean()*100:.1f}%)")
print(f"Accuracy (Before):      {df_before['is_correct'].mean()*100:.1f}% ({df_before['is_correct'].sum()}/{len(df_before)})")
print(f"Accuracy (After):       {df_after['is_correct'].mean()*100:.1f}% ({df_after['is_correct'].sum()}/{len(df_after)})")

short_b = df_before[df_before['clip_length_frames'] < 64]
short_a = df_after[df_after['clip_length_frames'] < 64]
print("\n--- Short Clips (< 64 frames) ---")
print(f"Short clips count:      {len(short_b)}")
print(f"Short missing (Before): {short_b['is_missing'].sum()}/{len(short_b)} ({short_b['is_missing'].mean()*100:.1f}%)")
print(f"Short missing (After):  {short_a['is_missing'].sum()}/{len(short_a)} ({short_a['is_missing'].mean()*100:.1f}%)")
print(f"Short correct (Before): {short_b['is_correct'].sum()}/{len(short_b)} ({short_b['is_correct'].mean()*100:.1f}%)")
print(f"Short correct (After):  {short_a['is_correct'].sum()}/{len(short_a)} ({short_a['is_correct'].mean()*100:.1f}%)")

long_b = df_before[df_before['clip_length_frames'] >= 64]
long_a = df_after[df_after['clip_length_frames'] >= 64]
print("\n--- Long Clips (>= 64 frames) ---")
print(f"Long clips count:       {len(long_b)}")
print(f"Long missing (Before):  {long_b['is_missing'].sum()}/{len(long_b)} ({long_b['is_missing'].mean()*100:.1f}%)")
print(f"Long missing (After):   {long_a['is_missing'].sum()}/{len(long_a)} ({long_a['is_missing'].mean()*100:.1f}%)")
print(f"Long correct (Before):  {long_b['is_correct'].sum()}/{len(long_b)} ({long_b['is_correct'].mean()*100:.1f}%)")
print(f"Long correct (After):   {long_a['is_correct'].sum()}/{len(long_a)} ({long_a['is_correct'].mean()*100:.1f}%)")

valid_b = df_before[df_before['first_stable_frame'] != -1]
valid_a = df_after[df_after['first_stable_frame'] != -1]
print("\n--- Latency to First Stable Prediction ---")
print(f"Evaluated clips with stable pred (Before): {len(valid_b)} / {len(df_before)}")
print(f"Evaluated clips with stable pred (After):  {len(valid_a)} / {len(df_after)}")
print(f"Mean first stable frame (Before): {valid_b['first_stable_frame'].mean():.1f} frames ({valid_b['first_stable_time_sec'].mean():.2f}s)")
print(f"Mean first stable frame (After):  {valid_a['first_stable_frame'].mean():.1f} frames ({valid_a['first_stable_time_sec'].mean():.2f}s)")
print(f"Median first stable frame (Before): {valid_b['first_stable_frame'].median():.1f} frames ({valid_b['first_stable_time_sec'].median():.2f}s)")
print(f"Median first stable frame (After):  {valid_a['first_stable_frame'].median():.1f} frames ({valid_a['first_stable_time_sec'].median():.2f}s)")

print("\n--- Error Reason Breakdown ---")
print("BEFORE:")
print(df_before['error_reason'].value_counts())
print("\nAFTER:")
print(df_after['error_reason'].value_counts())
