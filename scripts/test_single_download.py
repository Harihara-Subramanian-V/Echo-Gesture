"""
Test yt-dlp client options (android, ios, mweb) for YouTube download.
"""

import os
import subprocess
import yt_dlp
from yt_dlp.utils import download_range_func

url = "https://www.youtube.com/watch?v=J7tP98oDxqE"
start_time = 0.0
end_time = 2.64
test_out = r"c:\EchoCam\data\raw\ms_asl\test_sample.mp4"

# Try with android client which doesn't trigger the web bot verification
ydl_opts = {
    "format": "best[ext=mp4]/best",
    "outtmpl": test_out,
    "download_ranges": download_range_func(None, [(start_time, end_time)]),
    "force_keyframes_at_cuts": True,
    "extractor_args": {
        "youtube": {
            "player_client": ["android", "ios", "mweb"]
        }
    },
    "quiet": False
}

print("Testing download with android/ios/mweb player_client...")
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Download finished!")
    if os.path.exists(test_out):
        size = os.path.getsize(test_out)
        print(f"File size: {size} bytes")
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", test_out]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"ffprobe duration: {res.stdout.strip()}s (target: {end_time - start_time:.2f}s)")
except Exception as e:
    print(f"Error: {e}")
