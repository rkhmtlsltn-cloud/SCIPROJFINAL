import pandas as pd
import requests
import time

print("loading parquet...")

df = pd.read_parquet("ml_data.parquet")

stations = df[["station_name", "latitude", "longitude", "year"]].drop_duplicates().copy()

print(df.shape)
print("unique stations:", len(stations))

url = "https://ic.imagery1.arcgis.com/arcgis/rest/services/Sentinel2_10m_LandCover/ImageServer/identify"

landcover_values = []

for i, row in stations.iterrows():
    lat = row["latitude"]
    lon = row["longitude"]
    year = int(row["year"])

    params = {
        "f": "json",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "sr": 4326,
        "time": str(year),
        "returnGeometry": "false"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        print(data)
        value = data.get("value")
        landcover_values.append(value)

    except Exception:
        landcover_values.append(None)

    print("done:", len(landcover_values), "/", len(stations))

    time.sleep(0.01)

stations["landcover"] = landcover_values

df = df.merge(
    stations,
    on=["station_name", "latitude", "longitude", "year"],
    how="left"
)

print(df["landcover"].value_counts(dropna=False))

df.to_parquet("ml_data_sentinel.parquet", index=False)

print("saved ml_data_sentinel.parquet")