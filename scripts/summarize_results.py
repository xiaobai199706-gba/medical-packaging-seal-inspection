"""Summarize published CSV values; no model loading or prediction evaluation."""
from pathlib import Path
import argparse,csv,statistics

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-directory",type=Path,default=Path(__file__).resolve().parents[1]/"results")
    args=parser.parse_args()
    sources=[("Official P01",read(args.results_directory/"official/p01_supervision.csv"),"dice"),
             ("Content-group (20 validation defects)",[r for r in read(args.results_directory/"grouped/per_seed_results.csv") if r["calibration_defects"]=="20"],"pixel_dice")]
    for title,rows,dice in sources:
        print(title)
        for mode in dict.fromkeys(r["mode"] for r in rows):
            selected=[r for r in rows if r["mode"]==mode]
            values=[]
            for key in ["image_auroc","pixel_ap",dice]:
                x=[float(r[key]) for r in selected]
                values.append(f"{key}={statistics.mean(x):.6f} +/- {statistics.stdev(x):.6f}")
            print(mode+": "+"; ".join(values))

if __name__=="__main__":main()
