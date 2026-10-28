from pathlib import Path
import csv

path = Path("data/jisdor.csv")

if not path.exists():
    raise FileNotFoundError(f"File tidak ditemukan: {path.resolve()}")

with open(path, "r", encoding="utf-8-sig", newline="") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break

        line = line.rstrip("\r\n")

        print(f"\nBaris {i + 1} mentah:")
        print(repr(line))

        for separator in [",", ";"]:
            print(f"Pemisah {separator!r}, pembacaan asli:")
            print(next(csv.reader([line], delimiter=separator)))

            # Uji apabila seluruh baris dibungkus tanda kutip
            unwrapped = line.strip()

            if (
                len(unwrapped) >= 2
                and unwrapped.startswith('"')
                and unwrapped.endswith('"')
            ):
                unwrapped = unwrapped[1:-1]

            print(f"Pemisah {separator!r}, setelah uji kutip:")
            print(next(csv.reader([unwrapped], delimiter=separator)))