from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/raw/dataset.csv"; MODEL_DIR=ROOT/"models"; MODEL_DIR.mkdir(exist_ok=True)
if not DATA.exists(): raise FileNotFoundError("Put the car-price CSV at data/raw/dataset.csv")
df=pd.read_csv(DATA); lower={c.lower():c for c in df.columns}
target=next((lower[k] for k in ["price","selling_price","sellingprice","msrp"] if k in lower),None)
if target is None: raise ValueError(f"No price target found. Columns: {df.columns.tolist()}")
df[target]=pd.to_numeric(df[target],errors="coerce"); df=df.dropna(subset=[target])
X=df.drop(columns=[target]); y=df[target]
X=X.drop(columns=[c for c in X.columns if c.lower() in {"id","car_id"}],errors="ignore")
num=X.select_dtypes(include=np.number).columns.tolist(); cat=X.select_dtypes(exclude=np.number).columns.tolist()
pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),num),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42)
models={"Linear Regression":LinearRegression(),"Random Forest":RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1),"Gradient Boosting":GradientBoostingRegressor(random_state=42)}
for name,est in models.items():
    pipe=Pipeline([("preprocess",pre),("model",est)]); pipe.fit(Xt,yt); pred=pipe.predict(Xv)
    print(name,{"MAE":mean_absolute_error(yv,pred),"RMSE":mean_squared_error(yv,pred)**0.5,"R2":r2_score(yv,pred)})
final=Pipeline([("preprocess",pre),("model",models["Gradient Boosting"])]); final.fit(Xt,yt); dump(final,MODEL_DIR/"model.joblib")
