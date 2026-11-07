"""Recreate the House Edge forecast CSV. Python 3.11+ standard library only.

python predict.py --data /data --out /output/submission.csv            # test split (default)
python predict.py --data public --out val.csv --split validation       # local scoring
"""
import argparse
from pathlib import Path

import xorcast

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="test")
a = ap.parse_args()

model = xorcast.load_model(Path(__file__).resolve().parent / "model.json")
split = Path(a.data) / a.split
queries = xorcast.read_queries(split / "queries.csv")
history = xorcast.read_history(split / "history.csv")
xorcast.write_submission(a.out, xorcast.predict(queries, history, model))
print(f"wrote {len(queries)} predictions to {a.out}")
