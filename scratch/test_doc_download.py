import requests
from pathlib import Path

url = "https://fit.hcmuaf.edu.vn/data/file/CauTrucDuLieu.doc"
print(f"Downloading {url}...")
resp = requests.get(url, verify=False)
if resp.status_code == 200:
    print(f"Success! File size: {len(resp.content)} bytes")
    # Ghi thử ra file rác
    out_path = Path("scratch/test_ctdl.doc")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    print("Saved to scratch/test_ctdl.doc")
else:
    print(f"Failed: {resp.status_code}")
