"""Read original Feather tables and inspect schema before choosing ID columns."""
from pathlib import Path
import pandas as pd
import pyarrow.feather as feather

root = Path(__file__).resolve().parents[1] / "data" / "malecns"
for path in sorted(root.glob("*.feather")):
    table = feather.read_table(path, memory_map=True)
    table.validate(full=True)
    frame = table.to_pandas()
    print(f"\nFILE: {path.name}\nshape: {frame.shape}")
    print("columns:", frame.columns.tolist())
    print("dtypes:\n", frame.dtypes.to_string())
    print("first rows:\n", frame.head(3).to_string(index=False))
    print("null counts:\n", frame.isna().sum().to_string())
    del frame, table
