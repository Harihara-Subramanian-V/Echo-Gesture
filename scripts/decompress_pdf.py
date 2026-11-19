"""
Decompress and read PDF text streams.
"""

import zlib
import re

pdf_path = r"c:\EchoCam\data\metadata\ms_asl\MS-ASL\C-UDA-0.1_annotated_discussion.pdf"

with open(pdf_path, "rb") as f:
    data = f.read()

# Find all streams
stream_pattern = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
streams = stream_pattern.findall(data)

all_text = []
for idx, s in enumerate(streams):
    try:
        decomp = zlib.decompress(s)
        # extract text within parentheses or brackets
        t_matches = re.findall(rb"\((.*?)\)", decomp)
        for tm in t_matches:
            dec = tm.decode("latin1", errors="ignore").strip()
            if len(dec) > 3 and not dec.startswith("/"):
                all_text.append(dec)
    except Exception:
        pass

print(f"Extracted {len(all_text)} text fragments from decompressed PDF:")
for idx, t in enumerate(all_text[:30]):
    print(f"  {idx+1}. {t}")
