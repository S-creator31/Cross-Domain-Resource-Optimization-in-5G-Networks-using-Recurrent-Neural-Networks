# data_collector.py
# Collects REAL network metrics from your machine using system tools.
# Saves them as a CSV that can be used to train or test the LSTM model.
#
# Run this script for a few minutes to collect a dataset.
# The longer you run it, the more data you get.

import subprocess
import psutil
import time
import csv
import os
import re
import platform
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_FILE    = "data/real_network_data.csv"
PING_HOST      = "8.8.8.8"      # Google DNS — reliable ping target
INTERVAL_SEC   = 2              # collect one reading every 2 seconds
TOTAL_READINGS = 500            # collect 500 readings (~17 minutes)
                                # increase for more training data

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_signal_strength_windows():
    """
    Get WiFi signal strength on Windows using netsh.
    Returns value in dBm (negative number like -65).
    """
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, text=True
        )
        for line in result.stdout.split("\n"):
            if "Signal" in line:
                pct = int(re.search(r"(\d+)%", line).group(1))
                # Convert percentage to dBm approximation
                # 100% ≈ -30 dBm, 0% ≈ -90 dBm
                dbm = (pct / 2) - 100
                return round(dbm, 1)
    except Exception as e:
        print(f"  Signal error: {e}")
    return -70.0  # fallback default


def get_latency_ms():
    """
    Ping 8.8.8.8 and return average latency in ms.
    Works on both Windows and Linux/Mac.
    """
    try:
        if platform.system() == "Windows":
            cmd = ["ping", "-n", "3", PING_HOST]
        else:
            cmd = ["ping", "-c", "3", PING_HOST]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stdout

        # Windows: "Average = 23ms"
        win_match = re.search(r"Average = (\d+)ms", output)
        if win_match:
            return float(win_match.group(1))

        # Linux/Mac: "rtt min/avg/max/mdev = 12.3/15.6/20.1/2.3 ms"
        unix_match = re.search(r"[\d.]+/([\d.]+)/[\d.]+/[\d.]+ ms", output)
        if unix_match:
            return float(unix_match.group(1))

    except Exception as e:
        print(f"  Ping error: {e}")
    return 30.0  # fallback default


def get_bandwidth_mbps():
    """
    Get current network send/receive rates using psutil.
    Returns (bytes_sent_rate_mbps, bytes_recv_rate_mbps) over 1 second.
    """
    try:
        net1 = psutil.net_io_counters()
        time.sleep(1)
        net2 = psutil.net_io_counters()

        sent_mbps = ((net2.bytes_sent - net1.bytes_sent) * 8) / 1_000_000
        recv_mbps = ((net2.bytes_recv - net1.bytes_recv) * 8) / 1_000_000

        return round(sent_mbps, 3), round(recv_mbps, 3)
    except Exception as e:
        print(f"  Bandwidth error: {e}")
    return 1.0, 1.0


def get_top_network_app():
    """
    Find which application is using the most network bandwidth right now.
    Maps it to one of the 5 application types used in training.
    """
    app_map = {
        # Browser processes → Browsing
        "chrome": "Browsing", "firefox": "Browsing",
        "msedge": "Browsing", "opera": "Browsing",
        # Streaming/video apps
        "vlc": "Streaming", "netflix": "Streaming",
        "youtube": "Streaming", "spotify": "Streaming",
        "teams": "Streaming", "zoom": "Streaming",
        # Gaming
        "steam": "Gaming", "epicgameslauncher": "Gaming",
        "leagueclient": "Gaming", "minecraft": "Gaming",
        # VoIP / calls
        "discord": "VoIP", "skype": "VoIP", "whatsapp": "VoIP",
        # File transfer / downloads
        "utorrent": "File Transfer", "bittorrent": "File Transfer",
        "onedrive": "File Transfer", "dropbox": "File Transfer",
    }

    try:
        connections = psutil.net_connections(kind="inet")
        active_pids = set(c.pid for c in connections if c.pid)

        for pid in active_pids:
            try:
                proc = psutil.Process(pid)
                name = proc.name().lower().replace(".exe", "")
                for key, app_type in app_map.items():
                    if key in name:
                        return app_type
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        print(f"  App detection error: {e}")

    return "Browsing"  # default fallback


def estimate_resource_allocation(signal, latency, alloc_bw, req_bw):
    """
    Estimate resource allocation % from real metrics.
    This is the TARGET variable — approximated from actual conditions.
    Formula: based on how much of required bandwidth was actually allocated,
    adjusted by signal quality penalty.
    """
    if req_bw == 0:
        req_bw = 0.1

    # Base allocation: ratio of allocated to required
    base = min(alloc_bw / req_bw, 1.0)

    # Latency penalty: high latency = lower effective allocation
    latency_penalty = max(0, 1 - (latency / 200))

    # Signal penalty: weak signal = lower effective allocation
    signal_score = max(0, (signal + 90) / 60)  # -90dBm=0, -30dBm=1

    allocation = base * 0.6 + latency_penalty * 0.2 + signal_score * 0.2
    return round(min(max(allocation, 0.05), 1.0), 4)


# ── Main collection loop ──────────────────────────────────────────────────────

def collect_data():
    os.makedirs("data", exist_ok=True)

    fieldnames = [
        "Timestamp", "Signal_Strength", "Latency",
        "Required_Bandwidth", "Allocated_Bandwidth",
        "Application_Type", "Resource_Allocation"
    ]

    # Check if file exists — append if yes, create fresh if no
    file_exists = os.path.exists(OUTPUT_FILE)
    mode = "a" if file_exists else "w"

    print("="*55)
    print("Real-Time 5G Network Data Collector")
    print("="*55)
    print(f"Output file  : {OUTPUT_FILE}")
    print(f"Interval     : {INTERVAL_SEC} seconds per reading")
    print(f"Total target : {TOTAL_READINGS} readings")
    print(f"Est. time    : ~{(TOTAL_READINGS * INTERVAL_SEC) // 60} minutes")
    print("\nPress Ctrl+C to stop early (data collected so far is saved).")
    print("="*55)
    time.sleep(2)

    count = 0
    with open(OUTPUT_FILE, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        while count < TOTAL_READINGS:
            try:
                print(f"\nReading {count+1}/{TOTAL_READINGS}...")

                # Collect all metrics
                signal   = get_signal_strength_windows()
                latency  = get_latency_ms()
                sent_bw, recv_bw = get_bandwidth_mbps()
                app_type = get_top_network_app()

                # Required = what was received, Allocated = what was sent+received
                req_bw   = round(recv_bw + 0.5, 3)   # slightly more than received
                alloc_bw = round(recv_bw, 3)

                # Compute target variable
                resource_alloc = estimate_resource_allocation(
                    signal, latency, alloc_bw, req_bw
                )

                row = {
                    "Timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Signal_Strength":     f"{signal} dBm",
                    "Latency":             f"{latency} ms",
                    "Required_Bandwidth":  f"{req_bw} Mbps",
                    "Allocated_Bandwidth": f"{alloc_bw} Mbps",
                    "Application_Type":    app_type,
                    "Resource_Allocation": f"{round(resource_alloc * 100, 1)}%"
                }

                writer.writerow(row)
                f.flush()  # save immediately so data isn't lost on Ctrl+C

                # Print summary
                print(f"  Signal     : {signal} dBm")
                print(f"  Latency    : {latency} ms")
                print(f"  Bandwidth  : {alloc_bw} Mbps")
                print(f"  App Type   : {app_type}")
                print(f"  Resource % : {resource_alloc * 100:.1f}%")

                count += 1
                time.sleep(max(0, INTERVAL_SEC - 1))  # -1 for the 1s in bandwidth

            except KeyboardInterrupt:
                print(f"\n\nStopped early. {count} readings saved to {OUTPUT_FILE}")
                return

    print(f"\nDone! {count} readings saved to {OUTPUT_FILE}")
    print("You can now use this file to train or test your LSTM model.")


if __name__ == "__main__":
    collect_data()