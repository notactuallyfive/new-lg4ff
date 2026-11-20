import argparse
import csv
import glob
import sys
import numpy as np


def get_sysfs_path():
    paths = glob.glob("/sys/bus/hid/drivers/logitech/*/ffb_lut")
    if not paths:
        print("Error: ffb_lut sysfs node not found.", file=sys.stderr)
        sys.exit(1)
    return paths[0]


def build_rasmusp_lut(strength=0.6):
    # RasmusP G25/G27/G29 NoClipButStronger / RealNew80 smooth curve
    ctrl_in  = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.35, 0.54, 0.75, 1.00])
    ctrl_out = np.array([0.10, 0.12, 0.16, 0.19, 0.22, 0.24, 0.33, 0.46, 0.62, 0.78, 0.91, 1.00])

    # Blend between 1:1 linear (0.0) and full RasmusP (1.0)
    blended_out = ctrl_in + strength * (ctrl_out - ctrl_in)
    x = np.linspace(0.0, 1.0, 256)
    return np.interp(x, ctrl_in, blended_out)


def build_wheelcheck_lut(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))

    x_cmd = np.array([float(r["force"]) for r in rows]) / 10000.0
    y_resp = np.array([float(r["deltaX"]) for r in rows])
    y_resp = np.maximum.accumulate(y_resp / y_resp.max())

    # Invert measured response onto 256 uniform target forces
    target = np.linspace(0.0, 1.0, 256)
    inv = np.interp(target, y_resp, x_cmd)

    # Smooth staircase steps from encoder/gear friction
    kernel = np.ones(7) / 7.0
    return np.convolve(np.pad(inv, 3, mode="edge"), kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser(description="Load or clear G27 FFB LUT.")
    parser.add_argument(
        "-c", "--clear",
        action="store_true",
        help="Disable/clear the active LUT in the kernel driver",
    )
    parser.add_argument(
        "-r", "--rasmusp",
        action="store_true",
        help="Use RasmusP's smooth G27 curve instead of inverting a WheelCheck CSV",
    )
    parser.add_argument(
        "-s", "--strength",
        type=float,
        default=0.6,
        help="RasmusP curve blend strength from 0.0 (linear) to 1.0 (full RasmusP) (default: %(default)s)",
    )
    parser.add_argument(
        "--csv",
        default="wheelcheck.csv",
        help="Path to WheelCheck CSV file (default: %(default)s)",
    )
    parser.add_argument(
        "-t", "--taper",
        type=float,
        default=None,
        help="Center taper bin width (default: 0 for --rasmusp, 3.0 for WheelCheck CSV)",
    )
    args = parser.parse_args()

    path = get_sysfs_path()

    if args.clear:
        with open(path, "w") as f:
            f.write("disabled\n")
        print(f"Cleared/disabled LUT at: {path}")
        return

    if args.rasmusp:
        strength = max(0.0, min(1.0, args.strength))
        curve = build_rasmusp_lut(strength=strength)
        taper_bins = 0.0 if args.taper is None else args.taper
        mode_label = f"RasmusP smooth G27 curve (strength={strength})"
    else:
        curve = build_wheelcheck_lut(args.csv)
        taper_bins = 3.0 if args.taper is None else args.taper
        mode_label = f"WheelCheck CSV ({args.csv})"

    if taper_bins > 0:
        taper = np.minimum(1.0, np.arange(256) / taper_bins)
        curve = curve * taper

    lut = np.round(curve * 65535).astype(int)
    lut[0] = 0
    lut[-1] = 65535

    with open(path, "w") as f:
        f.write(" ".join(map(str, lut)) + "\n")

    print(f"Loaded {mode_label} (taper={taper_bins}) into: {path}")
    print(lut)


if __name__ == "__main__":
    main()
