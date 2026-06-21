import time
import subprocess
import os

print("Syncing training_status.json from Modal volume 'protean-models' to local workspace root...")
print("Press Ctrl+C to stop.")

local_file = "training_status.json"

while True:
    try:
        # Fetch the training status file from the remote volume
        res = subprocess.run(
            ["modal", "volume", "get", "protean-models", "training_status.json", "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if res.returncode == 0:
            print(".", end="", flush=True)
        else:
            # If the file doesn't exist yet on the volume, it will output an error. We skip it silently.
            if "not found" in res.stderr.lower():
                print("?", end="", flush=True)
            else:
                print("x", end="", flush=True)
    except Exception as e:
        print(f"\n[sync] Error: {e}")
    time.sleep(5)
