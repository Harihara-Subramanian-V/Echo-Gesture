"""
Extract and analyze license terms from C-UDA-0.1_annotated_discussion.pdf.
"""

import os
import re

pdf_path = r"c:\EchoCam\data\metadata\ms_asl\MS-ASL\C-UDA-0.1_annotated_discussion.pdf"

with open(pdf_path, "rb") as f:
    raw_data = f.read()

# Match text patterns
text_blocks = re.findall(rb"[a-zA-Z0-9 ,.;:()\'\"/\-\n]{10,}", raw_data)
decoded = [b.decode("latin1", errors="ignore").strip() for b in text_blocks]
decoded = [t for t in decoded if len(t) > 25]

print(f"Total text blocks extracted: {len(decoded)}")
print("\nSample Key Excerpts from C-UDA 0.1 PDF:")
for idx, d in enumerate(decoded[:25]):
    print(f"[{idx+1}] {d}")
