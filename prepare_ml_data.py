import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("start")

excel_path = os.path.join(BASE_DIR, "infosci_filled_by_years.xlsx")
print("reading excel...")

needed_names = {
    "datetime", "date", "time", "timestamp", "measured_at", "measuredat",
    "name", "station_name",
    "pm25", "pm_25", "pm2.5", "pm2_5",
    "lat", "latitude",
    "lon", "lng", "longitude",
    "district",
    "district_ru", "district_rus", "districtru"
}

def clean_col(c):
    return str(c).strip().lower()

def use_column(c):
    c = clean_col(c)
    if c in needed_names:
        return True
    if "district" in c and "ru" in c:
        return True
    if "pm25" in c or "pm2.5" in c:
        return True
    return False

excel_file = pd.ExcelFile(excel_path, engine="openpyxl")

frames = []

for sheet_name in excel_file.sheet_names:
    print("reading sheet:", sheet_name)

    sheet_df = pd.read_excel(
        excel_file,
        sheet_name=sheet_name,
        engine="openpyxl",
        usecols=use_column,
        dtype=str
    )

    sheet_df.columns = [str(c).strip() for c in sheet_df.columns]

    if "datetime" in sheet_df.columns:
        sheet_df = sheet_df[sheet_df["datetime"].astype(str).str.lower() != "datetime"]

    sheet_df["excel_sheet"] = str(sheet_name)
    frames.append(sheet_df)

stations_df = pd.concat(frames, ignore_index=True)

print("excel loaded")
print("rows before cleaning:", len(stations_df))

stations_df.columns = [str(c).strip() for c in stations_df.columns]

col_map = {}
for c in stations_df.columns:
    col_map[str(c).strip().lower()] = c

def find_col(keys):
    for key in keys:
        if key in col_map:
            return col_map[key]
    return None

name_col = find_col(["name", "station_name"])
pm25_col = find_col(["pm25", "pm_25", "pm2.5", "pm2_5"])
lat_col = find_col(["lat", "latitude"])
lon_col = find_col(["lon", "lng", "longitude"])
district_col = find_col(["district"])
district_ru_col = find_col(["district_ru", "district_rus", "districtru"])
date_col = find_col(["date", "datetime", "time", "timestamp", "measured_at", "measuredat"])

if district_ru_col is None:
    for c in stations_df.columns:
        low = str(c).strip().lower()
        if "district" in low and "ru" in low:
            district_ru_col = c
            break

if pm25_col is None:
    for c in stations_df.columns:
        low = str(c).strip().lower()
        if "pm25" in low or "pm2.5" in low:
            pm25_col = c
            break

need_cols = [name_col, pm25_col, lat_col, lon_col, district_col, district_ru_col, date_col]

if any(col is None for col in need_cols):
    print("columns found:", stations_df.columns.tolist())
    print("name_col =", name_col)
    print("pm25_col =", pm25_col)
    print("lat_col =", lat_col)
    print("lon_col =", lon_col)
    print("district_col =", district_col)
    print("district_ru_col =", district_ru_col)
    print("date_col =", date_col)
    raise ValueError("Не удалось определить нужные колонки в Excel")

stations_df = stations_df[
    [name_col, pm25_col, lat_col, lon_col, district_col, district_ru_col, date_col]
].copy()

stations_df.columns = [
    "station_name",
    "pm25",
    "latitude",
    "longitude",
    "district",
    "district_ru",
    "date"
]

stations_df["type"] = "Air Station"

for col in ["latitude", "longitude", "pm25"]:
    stations_df[col] = (
        stations_df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    stations_df[col] = pd.to_numeric(stations_df[col], errors="coerce")

for col in ["district_ru", "district", "station_name", "date"]:
    stations_df[col] = stations_df[col].astype(str).str.strip()

stations_df["date_parsed"] = pd.to_datetime(
    stations_df["date"],
    format="%d.%m.%Y %H:%M",
    errors="coerce"
)

mask_bad = stations_df["date_parsed"].isna()

if mask_bad.sum() > 0:
    print("dates not parsed in first format:", int(mask_bad.sum()))
    stations_df.loc[mask_bad, "date_parsed"] = pd.to_datetime(
        stations_df.loc[mask_bad, "date"],
        errors="coerce",
        dayfirst=True
    )

stations_df["year"] = stations_df["date_parsed"].dt.year

stations_df = stations_df.dropna(
    subset=["latitude", "longitude", "district_ru", "date_parsed", "year"]
)

stations_df["year"] = stations_df["year"].astype(int)

print("rows after cleaning:", len(stations_df))
print("years found:", sorted(stations_df["year"].unique().tolist()))

ml_df = stations_df.copy()

ml_df["month"] = ml_df["date_parsed"].dt.month
ml_df["day"] = ml_df["date_parsed"].dt.day
ml_df["hour"] = ml_df["date_parsed"].dt.hour
ml_df["dayofweek"] = ml_df["date_parsed"].dt.dayofweek

ml_df = ml_df.sort_values(["station_name", "date_parsed"])

ml_df["pm25_prev_1"] = ml_df.groupby("station_name")["pm25"].shift(1)

ml_df["pm25_future_1"] = (
    ml_df.groupby("station_name")["pm25"].shift(-1)
)

ml_df["pm25_rolling_3"] = (
    ml_df.groupby("station_name")["pm25"]
    .transform(lambda x: x.rolling(3).mean())
)

ml_df["pm25_rolling_6"] = (
    ml_df.groupby("station_name")["pm25"]
    .transform(lambda x: x.rolling(6).mean())
)

ml_df = ml_df.dropna()

ml_df = ml_df.sort_values("date_parsed")

print("saving parquet...")

ml_df.to_parquet("ml_data.parquet", index=False)

print("ML data saved")
print(ml_df.shape)
print(ml_df.head())

