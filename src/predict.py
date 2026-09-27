from pathlib import Path
import pandas as pd
from joblib import load
ROOT=Path(__file__).resolve().parents[1]; model=load(ROOT/"models/model.joblib")
df=pd.read_csv(ROOT/"data/raw/sample.csv"); print(model.predict(df))
