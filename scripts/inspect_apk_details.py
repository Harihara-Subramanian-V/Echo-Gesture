import zipfile
import os
import sys

def inspect_apk(apk_path):
    if not os.path.exists(apk_path):
        print(f"File not found: {apk_path}")
        return

    apk_file_size = os.path.getsize(apk_path)
    with zipfile.ZipFile(apk_path, "r") as z:
        categories = {}
        so_files = {}
        assets = {}
        dex_files = {}
        total_comp = 0
        total_uncomp = 0
        
        for info in z.infolist():
            fn = info.filename
            comp = info.compress_size
            uncomp = info.file_size
            total_comp += comp
            total_uncomp += uncomp
            
            parts = fn.split("/")
            prefix = parts[0]
            if fn.endswith(".dex"):
                cat = "classes.dex"
                dex_files[fn] = (comp, uncomp)
            elif prefix == "lib":
                cat = "lib"
                if len(parts) >= 2:
                    abi = parts[1]
                    so_name = parts[-1]
                    if abi not in so_files:
                        so_files[abi] = []
                    so_files[abi].append((so_name, comp, uncomp, info.compress_type))
            elif prefix == "assets":
                cat = "assets"
                assets[fn] = (comp, uncomp, info.compress_type)
            elif prefix == "res":
                cat = "res"
            elif fn == "resources.arsc":
                cat = "resources.arsc"
            elif prefix == "META-INF":
                cat = "META-INF"
            else:
                cat = prefix
                
            if cat not in categories:
                categories[cat] = [0, 0, 0]
            categories[cat][0] += comp
            categories[cat][1] += uncomp
            categories[cat][2] += 1

    print("=" * 80)
    print(f"APK PATH: {apk_path}")
    print(f"APK Total File Size: {apk_file_size:,} bytes ({apk_file_size / (1024*1024):.2f} MB)")
    print(f"APK Total Compressed Zip Entries: {total_comp:,} bytes ({total_comp / (1024*1024):.2f} MB)")
    print(f"APK Total Uncompressed Payload: {total_uncomp:,} bytes ({total_uncomp / (1024*1024):.2f} MB)")
    print("=" * 80)

    print("\n[1] CATEGORY BREAKDOWN:")
    print(f"{'Category':<20} | {'Count':<8} | {'Compressed':<16} | {'Uncompressed':<16} | {'% of APK'}")
    print("-" * 75)
    for cat, (c, u, cnt) in sorted(categories.items(), key=lambda x: x[1][0], reverse=True):
        pct = (c / total_comp) * 100
        c_str = f"{c/(1024*1024):.2f} MB ({c:,} B)"
        u_str = f"{u/(1024*1024):.2f} MB ({u:,} B)"
        print(f"{cat:<20} | {cnt:<8} | {c_str:<16} | {u_str:<16} | {pct:6.2f}%")

    print("\n[2] ASSETS BREAKDOWN:")
    print(f"{'Asset Filename':<45} | {'Storage Mode':<15} | {'Compressed':<16} | {'Uncompressed':<16}")
    print("-" * 95)
    for a, (c, u, ct) in sorted(assets.items(), key=lambda x: x[1][1], reverse=True):
        mode = "STORED (0)" if ct == 0 else f"DEFLATED ({ct})"
        c_str = f"{c/1024:.2f} KB ({c:,} B)"
        u_str = f"{u/1024:.2f} KB ({u:,} B)"
        print(f"{a:<45} | {mode:<15} | {c_str:<16} | {u_str:<16}")

    print("\n[3] DEX BREAKDOWN:")
    for d, (c, u) in sorted(dex_files.items(), key=lambda x: x[1][0], reverse=True):
        print(f"  {d:<20} Compressed: {c/(1024*1024):.2f} MB ({c:,} B), Uncompressed: {u/(1024*1024):.2f} MB ({u:,} B)")

    print("\n[4] NATIVE LIBRARIES (lib/) PER ABI:")
    for abi in sorted(so_files.keys()):
        abi_c = sum(x[1] for x in so_files[abi])
        abi_u = sum(x[2] for x in so_files[abi])
        pct_of_apk = (abi_c / total_comp) * 100
        print(f"\n  ABI: {abi} - {len(so_files[abi])} .so files | Compressed: {abi_c/(1024*1024):.2f} MB ({pct_of_apk:.1f}% of APK) | Uncompressed: {abi_u/(1024*1024):.2f} MB")
        print(f"    {'Library Name':<40} | {'Mode':<10} | {'Compressed':<16} | {'Uncompressed':<16}")
        print("    " + "-" * 85)
        for so_name, c, u, ct in sorted(so_files[abi], key=lambda x: x[1], reverse=True):
            mode = "STORED" if ct == 0 else f"DEFLATED"
            c_str = f"{c/(1024*1024):.2f} MB ({c:,} B)"
            u_str = f"{u/(1024*1024):.2f} MB ({u:,} B)"
            print(f"    {so_name:<40} | {mode:<10} | {c_str:<16} | {u_str:<16}")

if __name__ == "__main__":
    apk = sys.argv[1] if len(sys.argv) > 1 else "android/app/build/outputs/apk/debug/app-debug.apk"
    inspect_apk(apk)
