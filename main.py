import os
import json
from datetime import datetime
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("start")

excel_path = os.path.join(BASE_DIR, "infosci_filled_by_years.xlsx")
if not os.path.exists(excel_path):
    excel_path = os.path.join(BASE_DIR, "infosci.xlsx")

geojson_path = os.path.join(BASE_DIR, "almaty.geo.json")
output_path = os.path.join(BASE_DIR, "final_map.html")

print("reading excel:", excel_path)

needed_names = {
    "datetime", "date", "time", "timestamp", "measured_at", "measuredat",
    "name", "station_name",
    "pm25", "pm_25", "pm2.5", "pm2_5",
    "pm10", "pm_10", "pm1.0", "pm1_0",
    "no2", "no_2", "nitrogen_dioxide",
    "lat", "latitude",
    "lon", "lng", "longitude",
    "district",
    "district_ru", "district_rus", "districtru",
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
    if "pm10" in c or "pm1.0" in c:
        return True
    if "no2" in c or "nitrogen" in c:
        return True
    return False


def find_col(col_map, keys):
    for key in keys:
        if key in col_map:
            return col_map[key]
    return None


def find_fuzzy(columns, checks):
    for c in columns:
        low = clean_col(c)
        if any(check in low for check in checks):
            return c
    return None


excel_file = pd.ExcelFile(excel_path, engine="openpyxl")
frames = []

for sheet_name in excel_file.sheet_names:
    print("reading sheet:", sheet_name)

    sheet_df = pd.read_excel(
        excel_file,
        sheet_name=sheet_name,
        engine="openpyxl",
        usecols=use_column,
        dtype=str,
    )

    if sheet_df.empty:
        continue

    sheet_df.columns = [str(c).strip() for c in sheet_df.columns]

    for c in sheet_df.columns:
        if clean_col(c) == "datetime":
            sheet_df = sheet_df[sheet_df[c].astype(str).str.lower() != "datetime"]
            break

    sheet_df["excel_sheet"] = str(sheet_name)
    frames.append(sheet_df)

if not frames:
    raise ValueError("Не удалось найти нужные данные в Excel")

stations_df = pd.concat(frames, ignore_index=True)

print("excel loaded")
print("rows before cleaning:", len(stations_df))

stations_df.columns = [str(c).strip() for c in stations_df.columns]

col_map = {}
for c in stations_df.columns:
    col_map[clean_col(c)] = c

name_col = find_col(col_map, ["name", "station_name"])
pm25_col = find_col(col_map, ["pm25", "pm_25", "pm2.5", "pm2_5"])
pm10_col = find_col(col_map, ["pm10", "pm_10", "pm1.0", "pm1_0"])
no2_col = find_col(col_map, ["no2", "no_2", "nitrogen_dioxide"])
lat_col = find_col(col_map, ["lat", "latitude"])
lon_col = find_col(col_map, ["lon", "lng", "longitude"])
district_col = find_col(col_map, ["district"])
district_ru_col = find_col(col_map, ["district_ru", "district_rus", "districtru"])
date_col = find_col(col_map, ["date", "datetime", "time", "timestamp", "measured_at", "measuredat"])

if pm25_col is None:
    pm25_col = find_fuzzy(stations_df.columns, ["pm25", "pm2.5"])

if pm10_col is None:
    pm10_col = find_fuzzy(stations_df.columns, ["pm10", "pm1.0"])

if no2_col is None:
    no2_col = find_fuzzy(stations_df.columns, ["no2", "nitrogen"])

if district_ru_col is None:
    for c in stations_df.columns:
        low = clean_col(c)
        if "district" in low and "ru" in low:
            district_ru_col = c
            break

required_cols = [name_col, lat_col, lon_col, district_col, district_ru_col, date_col]

if pm25_col is None and pm10_col is None and no2_col is None:
    print("columns found:", stations_df.columns.tolist())
    raise ValueError("Не найдены колонки загрязнения: pm25, pm10 или no2")

if any(col is None for col in required_cols):
    print("columns found:", stations_df.columns.tolist())
    print("name_col =", name_col)
    print("pm25_col =", pm25_col)
    print("pm10_col =", pm10_col)
    print("no2_col =", no2_col)
    print("lat_col =", lat_col)
    print("lon_col =", lon_col)
    print("district_col =", district_col)
    print("district_ru_col =", district_ru_col)
    print("date_col =", date_col)
    raise ValueError("Не удалось определить нужные колонки в Excel")

selected_cols = [name_col, lat_col, lon_col, district_col, district_ru_col, date_col]
rename_map = {
    name_col: "station_name",
    lat_col: "latitude",
    lon_col: "longitude",
    district_col: "district",
    district_ru_col: "district_ru",
    date_col: "date",
}

if pm25_col is not None:
    selected_cols.append(pm25_col)
    rename_map[pm25_col] = "pm25"

if pm10_col is not None:
    selected_cols.append(pm10_col)
    rename_map[pm10_col] = "pm10"

if no2_col is not None:
    selected_cols.append(no2_col)
    rename_map[no2_col] = "no2"

stations_df = stations_df[selected_cols].copy()
stations_df = stations_df.rename(columns=rename_map)

for pollutant in ["pm25", "pm10", "no2"]:
    if pollutant not in stations_df.columns:
        stations_df[pollutant] = None

stations_df["type"] = "Air Station"

for col in ["latitude", "longitude", "pm25", "pm10", "no2"]:
    stations_df[col] = (
        stations_df[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .replace({"None": None, "nan": None, "": None})
    )
    stations_df[col] = pd.to_numeric(stations_df[col], errors="coerce")

for col in ["district_ru", "district", "station_name", "date"]:
    stations_df[col] = stations_df[col].astype(str).str.strip()

stations_df["date_parsed"] = pd.to_datetime(
    stations_df["date"],
    format="%d.%m.%Y %H:%M",
    errors="coerce",
)

mask_bad = stations_df["date_parsed"].isna()

if mask_bad.sum() > 0:
    print("dates not parsed in first format:", int(mask_bad.sum()))
    stations_df.loc[mask_bad, "date_parsed"] = pd.to_datetime(
        stations_df.loc[mask_bad, "date"],
        errors="coerce",
        dayfirst=True,
    )

stations_df["year"] = stations_df["date_parsed"].dt.year
stations_df["month"] = stations_df["date_parsed"].dt.month


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


stations_df["season"] = stations_df["month"].apply(get_season)

stations_df = stations_df.dropna(
    subset=["latitude", "longitude", "district_ru", "date_parsed", "year"]
)

stations_df["year"] = stations_df["year"].astype(int)

print("rows after cleaning:", len(stations_df))
print("years found:", sorted(stations_df["year"].unique().tolist()))

with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

print("geojson loaded")

for feature in geojson_data.get("features", []):
    props = feature.setdefault("properties", {})
    if "district" not in props or not props["district"]:
        ru_name = props.get("nameRu")
        if ru_name:
            props["district"] = ru_name

years = sorted(stations_df["year"].unique().tolist())
years_str = [str(y) for y in years]
seasons = ["Winter", "Spring", "Summer", "Autumn"]
pollutants = ["pm25", "pm10", "no2"]

district_info = {}

for (year, season, district_ru), group in stations_df.groupby(["year", "season", "district_ru"], sort=False):
    year = str(year)

    if year not in district_info:
        district_info[year] = {}

    if season not in district_info[year]:
        district_info[year][season] = {}

    pollution = {}
    for pollutant in pollutants:
        pollution[pollutant] = (
            round(group[pollutant].mean(), 2)
            if group[pollutant].notna().any()
            else None
        )

    district_info[year][season][district_ru] = {
        "district_ru": district_ru,
        "district_en": str(group["district"].iloc[0]),
        "pollution": pollution,
        "count": int(len(group)),
    }

stations_df = stations_df.sort_values("date_parsed")
stations_df["year"] = stations_df["year"].astype(str)
stations_df["date_text"] = stations_df["date_parsed"].dt.strftime("%d.%m.%Y %H:%M")

station_groups = {}

group_cols = ["year", "season", "district_ru", "station_name"]

for (year, season, district_ru, station_name), group in stations_df.groupby(group_cols, sort=False):
    group = group.sort_values("date_parsed")
    first_row = group.iloc[0]
    last_row = group.iloc[-1]

    measurements = []

    for _, row in group.iterrows():
        item = {"date": row["date_text"]}
        for pollutant in pollutants:
            item[pollutant] = (
                round(float(row[pollutant]), 2)
                if pd.notna(row[pollutant])
                else None
            )
        measurements.append(item)

    latest = {}
    for pollutant in pollutants:
        latest[pollutant] = (
            round(float(last_row[pollutant]), 2)
            if pd.notna(last_row[pollutant])
            else None
        )

    if year not in station_groups:
        station_groups[year] = {}

    if season not in station_groups[year]:
        station_groups[year][season] = {}

    if district_ru not in station_groups[year][season]:
        station_groups[year][season][district_ru] = []

    station_groups[year][season][district_ru].append({
        "station_name": str(station_name),
        "latitude": float(first_row["latitude"]),
        "longitude": float(first_row["longitude"]),
        "type": str(first_row["type"]),
        "district_ru": str(district_ru),
        "district_en": str(first_row["district"]),
        "year": str(year),
        "season": str(season),
        "latest": latest,
        "latest_date": str(last_row["date_text"]),
        "measurements": measurements,
    })

for year in station_groups:
    for season in station_groups[year]:
        for district_ru in station_groups[year][season]:
            station_groups[year][season][district_ru].sort(
                key=lambda x: x["station_name"].lower()
            )

print("stations with histories prepared")
print("data prepared")
print("years in final data:", years_str)

geojson_js = json.dumps(geojson_data, ensure_ascii=False)
district_info_js = json.dumps(district_info, ensure_ascii=False)
station_groups_js = json.dumps(station_groups, ensure_ascii=False)
years_js = json.dumps(years_str, ensure_ascii=False)
seasons_js = json.dumps(seasons, ensure_ascii=False)

build_time = datetime.now().strftime("%H:%M:%S")

html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Central Asia Land Cover Monitor {build_time}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css"/>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            font-family: Arial, sans-serif;
            background: #f6f7f9;
        }}

        #app {{
            display: flex;
            width: 100%;
            height: 100%;
        }}

        #map {{
            flex: 1;
            height: 100%;
            background: #eef2f6;
        }}

        #sidebar {{
            width: 390px;
            background: #ffffff;
            border-left: 1px solid #e5e7eb;
            padding: 22px 20px;
            box-sizing: border-box;
            overflow-y: auto;
            z-index: 1000;
        }}

        .title {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 8px;
            color: #111827;
        }}

        .subtitle {{
            font-size: 14px;
            color: #6b7280;
            margin-bottom: 18px;
        }}

        .card {{
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }}

        .district-name {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 12px;
            color: #111827;
        }}

        .metric {{
            margin-bottom: 10px;
            font-size: 15px;
            color: #1f2937;
        }}

        .metric b {{
            color: #111827;
        }}

        .stations-title {{
            font-size: 16px;
            font-weight: 700;
            margin: 0 0 10px;
            color: #111827;
        }}

        .station-item {{
            border-bottom: 1px solid #e5e7eb;
            padding: 10px 0;
            font-size: 14px;
            color: #374151;
        }}

        .station-item:last-child {{
            border-bottom: none;
        }}

        .legend {{
            margin-top: 8px;
            font-size: 14px;
            color: #374151;
        }}

        .legend-row {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}

        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
            border: 1px solid rgba(0,0,0,0.15);
        }}

        .square {{
            width: 13px;
            height: 13px;
            display: inline-block;
            margin-right: 10px;
            border-radius: 3px;
            border: 1px solid rgba(0,0,0,0.15);
        }}

        .hint {{
            font-size: 14px;
            color: #6b7280;
            line-height: 1.6;
        }}

        .top-btn {{
            margin-top: 10px;
            background: #0f172a;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 14px;
            cursor: pointer;
        }}

        .top-btn:hover {{
            background: #1e293b;
        }}

        .leaflet-tooltip {{
            font-size: 13px;
            padding: 6px 10px;
            border-radius: 8px;
            border: 1px solid #d1d5db;
            box-shadow: none;
        }}

        .select-box {{
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #d1d5db;
            font-size: 14px;
            background: white;
            margin-bottom: 10px;
        }}

        .leaflet-container {{
            font: inherit;
        }}

        .small-note {{
            font-size: 12px;
            color: #6b7280;
            line-height: 1.5;
            margin-top: 6px;
        }}

        @media (max-width: 800px) {{
            #app {{
                flex-direction: column;
            }}

            #map {{
                min-height: 58vh;
            }}

            #sidebar {{
                width: 100%;
                height: 42vh;
                border-left: none;
                border-top: 1px solid #e5e7eb;
            }}
        }}
    </style>
</head>
<body>
<div id="app">
    <div id="map"></div>

    <div id="sidebar">
        <div class="title">Central Asia Monitor</div>
        <div class="subtitle">OpenAQ-style map of stations + ArcGIS Land Cover</div>

        <div class="card">
            <div class="stations-title">Город</div>
            <select id="city-select" class="select-box">
                <option value="almaty">Алматы, KZ</option>
                <option value="astana">Астана, KZ</option>
                <option value="tashkent">Ташкент, UZ</option>
                <option value="samarkand">Самарканд, UZ</option>
                <option value="nukus">Нукус, UZ</option>
                <option value="bishkek">Бишкек, KG</option>
                <option value="talas">Талас, KG</option>
                <option value="dushanbe">Душанбе, TJ</option>
                <option value="ashgabat">Ашхабад, TM</option>
                <option value="turkmenbashi">Туркменбаши, TM</option>
            </select>

            <div class="stations-title">Год</div>
            <select id="year-select" class="select-box"></select>

            <div class="stations-title">Сезон</div>
            <select id="season-select" class="select-box"></select>

            <div class="stations-title">Показатель</div>
            <select id="pollutant-select" class="select-box">
                <option value="pm25">PM2.5</option>
                <option value="pm10">PM10</option>
                <option value="no2">NO2</option>
            </select>

            <div class="stations-title">Слой карты</div>
            <select id="layer-select" class="select-box">
                <option value="stations">Stations only</option>
                <option value="arcgis">Stations + ArcGIS Land Cover</option>
            </select>

            <div class="small-note">
                Станции сейчас есть только для Алматы. Для остальных городов доступен каркас и ArcGIS Land Cover.
            </div>
        </div>

        <div class="card" id="info-card">
            <div class="district-name">Алматы</div>
            <div class="hint">
                Нажми на район на карте.<br>
                Потом нажми на конкретную станцию.
            </div>
            <button class="top-btn" onclick="resetView()">Сбросить выбор</button>
        </div>

        <div class="card" id="pollution-legend-card">
            <div class="stations-title">Легенда загрязнения</div>
            <div class="legend">
                <div class="legend-row"><span class="dot" style="background:green;"></span>Меньше 50</div>
                <div class="legend-row"><span class="dot" style="background:orange;"></span>От 50 до 100</div>
                <div class="legend-row"><span class="dot" style="background:red;"></span>Больше 100</div>
                <div class="legend-row"><span class="dot" style="background:#9ca3af;"></span>Нет данных</div>
                <div class="legend-row"><span class="dot" style="background:#2563eb;"></span>Выбранный район</div>
            </div>
        </div>

        <div class="card" id="landcover-legend-card" style="display:none;">
            <div class="stations-title">Легенда Land Cover</div>
            <div class="legend">
                <div class="legend-row"><span class="square" style="background:#1A5BAB;"></span>Water</div>
                <div class="legend-row"><span class="square" style="background:#358221;"></span>Trees</div>
                <div class="legend-row"><span class="square" style="background:#87D19E;"></span>Flooded Vegetation</div>
                <div class="legend-row"><span class="square" style="background:#FFDB5C;"></span>Crops</div>
                <div class="legend-row"><span class="square" style="background:#ED022A;"></span>Built Area</div>
                <div class="legend-row"><span class="square" style="background:#EDE9E4;"></span>Bare Ground</div>
                <div class="legend-row"><span class="square" style="background:#F2FAFF;"></span>Snow/Ice</div>
                <div class="legend-row"><span class="square" style="background:#C8C8C8;"></span>Clouds</div>
                <div class="legend-row"><span class="square" style="background:#C6AD8D;"></span>Rangeland</div>
            </div>
        </div>

        <div class="card">
            <div class="stations-title">Станции / измерения</div>
            <div id="station-list" class="hint">Выбери район, потом станцию</div>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/esri-leaflet@3.0.12/dist/esri-leaflet.js"></script>
<script>
const geojsonData = {geojson_js};
const districtInfo = {district_info_js};
const stationGroups = {station_groups_js};
const years = {years_js};
const seasons = {seasons_js};

const pollutantLabels = {{
    pm25: "PM2.5",
    pm10: "PM10",
    no2: "NO2"
}};

const cityConfig = {{
    almaty: {{ name: "Алматы, Казахстан", center: [43.2389, 76.8897], zoom: 11, hasStations: true }},
    astana: {{ name: "Астана, Казахстан", center: [51.1694, 71.4491], zoom: 11, hasStations: false }},
    tashkent: {{ name: "Ташкент, Узбекистан", center: [41.2995, 69.2401], zoom: 11, hasStations: false }},
    samarkand: {{ name: "Самарканд, Узбекистан", center: [39.6542, 66.9597], zoom: 11, hasStations: false }},
    nukus: {{ name: "Нукус, Узбекистан", center: [42.4531, 59.6103], zoom: 11, hasStations: false }},
    bishkek: {{ name: "Бишкек, Кыргызстан", center: [42.8746, 74.5698], zoom: 11, hasStations: false }},
    talas: {{ name: "Талас, Кыргызстан", center: [42.5228, 72.2427], zoom: 12, hasStations: false }},
    dushanbe: {{ name: "Душанбе, Таджикистан", center: [38.5598, 68.7870], zoom: 11, hasStations: false }},
    ashgabat: {{ name: "Ашхабад, Туркменистан", center: [37.9601, 58.3261], zoom: 11, hasStations: false }},
    turkmenbashi: {{ name: "Туркменбаши, Туркменистан", center: [40.0222, 52.9552], zoom: 12, hasStations: false }}
}};

const map = L.map('map', {{ preferCanvas: false, zoomControl: true }}).setView([43.2389, 76.8897], 11);

L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

map.createPane('landCoverPane');
map.getPane('landCoverPane').style.zIndex = 250;

map.createPane('districtPane');
map.getPane('districtPane').style.zIndex = 400;

map.createPane('markerPane');
map.getPane('markerPane').style.zIndex = 650;

let currentCity = "almaty";
let currentMarkersLayer = L.layerGroup().addTo(map);
let selectedLayer = null;
let allDistrictLayers = [];
let geojsonLayer = null;
let selectedYear = years.length > 0 ? years[years.length - 1] : "2024";
let currentSeason = seasons.includes("Summer") ? "Summer" : seasons[0];
let currentPollutant = "pm25";
let currentMode = "stations";
let landCoverLayer = null;
let selectedDistrictRu = null;

const citySelect = document.getElementById('city-select');
const yearSelect = document.getElementById('year-select');
const seasonSelect = document.getElementById('season-select');
const pollutantSelect = document.getElementById('pollutant-select');
const layerSelect = document.getElementById('layer-select');
const landcoverLegendCard = document.getElementById('landcover-legend-card');
const pollutionLegendCard = document.getElementById('pollution-legend-card');

if (years.length === 0) {{
    const option = document.createElement('option');
    option.value = "2024";
    option.textContent = "2024";
    yearSelect.appendChild(option);
}} else {{
    for (let i = 0; i < years.length; i++) {{
        const option = document.createElement('option');
        option.value = years[i];
        option.textContent = years[i];
        yearSelect.appendChild(option);
    }}
}}

for (let i = 0; i < seasons.length; i++) {{
    const option = document.createElement('option');
    option.value = seasons[i];
    option.textContent = seasons[i];
    seasonSelect.appendChild(option);
}}

yearSelect.value = selectedYear;
seasonSelect.value = currentSeason;
pollutantSelect.value = currentPollutant;

citySelect.addEventListener('change', function() {{
    currentCity = this.value;
    changeCity();
}});

yearSelect.addEventListener('change', function() {{
    selectedYear = this.value;
    updateLandCover();
    resetView();
}});

seasonSelect.addEventListener('change', function() {{
    currentSeason = this.value;
    resetView();
}});

pollutantSelect.addEventListener('change', function() {{
    currentPollutant = this.value;
    if (selectedDistrictRu) {{
        updatePanel(selectedDistrictRu);
        showMarkersForDistrict(selectedDistrictRu);
    }} else {{
        resetView();
    }}
}});

layerSelect.addEventListener('change', function() {{
    currentMode = this.value;
    updateLandCover();
}});

function updateLandCover() {{
    if (landCoverLayer) {{
        map.removeLayer(landCoverLayer);
        landCoverLayer = null;
    }}

    if (currentMode === "arcgis" && typeof L.esri !== "undefined") {{
        landCoverLayer = L.esri.imageMapLayer({{
            url: "https://ic.imagery1.arcgis.com/arcgis/rest/services/Sentinel2_10m_LandCover/ImageServer",
            pane: "landCoverPane",
            opacity: 0.55,
            mosaicRule: {{ where: "Year=" + selectedYear }},
            renderingRule: {{ rasterFunction: "Cartographic Renderer for Visualization and Analysis" }}
        }}).addTo(map);

        landcoverLegendCard.style.display = "block";
    }} else {{
        landcoverLegendCard.style.display = "none";
    }}
}}

function getDefaultStyle() {{
    return {{ pane: 'districtPane', color: '#111827', weight: 2, opacity: 0.9, fillOpacity: 0 }};
}}

function getHoverStyle() {{
    return {{ pane: 'districtPane', color: '#374151', weight: 3, opacity: 1, fillOpacity: 0 }};
}}

function getHighlightStyle() {{
    return {{ pane: 'districtPane', color: '#2563eb', weight: 4, opacity: 1, fillOpacity: 0 }};
}}

function getColor(value) {{
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '#9ca3af';
    value = Number(value);
    if (value < 50) return 'green';
    if (value < 100) return 'orange';
    return 'red';
}}

function formatValue(value) {{
    if (value === null || value === undefined || Number.isNaN(Number(value))) return 'Нет данных';
    return Number(value).toFixed(2);
}}

function clearMarkers() {{
    currentMarkersLayer.clearLayers();
}}

function getStationsForDistrict(districtRu) {{
    return (
        stationGroups[selectedYear] &&
        stationGroups[selectedYear][currentSeason] &&
        stationGroups[selectedYear][currentSeason][districtRu]
    ) || [];
}}

function showStationDetails(station) {{
    const stationList = document.getElementById('station-list');

    if (!station || !station.measurements || station.measurements.length === 0) {{
        stationList.innerHTML = 'Нет измерений';
        return;
    }}

    let html = `
        <div class="station-item">
            <b>${{station.station_name}}</b><br>
            Район: ${{station.district_ru}}<br>
            Показатель: ${{pollutantLabels[currentPollutant]}}<br>
            Всего измерений: ${{station.measurements.length}}
        </div>
    `;

    for (let i = station.measurements.length - 1; i >= 0; i--) {{
        const m = station.measurements[i];
        html += `
            <div class="station-item">
                <b>${{m.date}}</b><br>
                ${{pollutantLabels[currentPollutant]}}: ${{formatValue(m[currentPollutant])}}
            </div>
        `;
    }}

    stationList.innerHTML = html;
}}

function showMarkersForDistrict(districtRu) {{
    clearMarkers();

    if (currentCity !== "almaty") {{
        return;
    }}

    const stations = getStationsForDistrict(districtRu);

    for (let i = 0; i < stations.length; i++) {{
        const s = stations[i];
        const value = s.latest ? s.latest[currentPollutant] : null;
        const color = getColor(value);

        const marker = L.circleMarker([s.latitude, s.longitude], {{
            pane: 'markerPane',
            radius: 7,
            color: color,
            fillColor: color,
            fillOpacity: 0.95,
            opacity: 1,
            weight: 2,
            interactive: true
        }});

        marker.bindPopup(
            `<b>${{s.station_name}}</b><br>${{pollutantLabels[currentPollutant]}}: ${{formatValue(value)}}<br>District: ${{s.district_ru}}<br>Last date: ${{s.latest_date}}<br>Year: ${{s.year}}<br>Season: ${{s.season}}`
        );

        marker.on('click', function() {{
            showStationDetails(s);
        }});

        currentMarkersLayer.addLayer(marker);
    }}
}}

function updatePanel(districtRu) {{
    selectedDistrictRu = districtRu;

    const data = (
        districtInfo[selectedYear] &&
        districtInfo[selectedYear][currentSeason] &&
        districtInfo[selectedYear][currentSeason][districtRu]
    ) || null;

    const panel = document.getElementById('info-card');
    const stationList = document.getElementById('station-list');

    if (currentCity !== "almaty") {{
        const cfg = cityConfig[currentCity];

        panel.innerHTML = `
            <div class="district-name">${{cfg.name}}</div>
            <div class="hint">
                Станций пока нет.<br>
                Сейчас доступен только каркас города и ArcGIS Land Cover слой.<br>
                Год: ${{selectedYear}}<br>
                Сезон: ${{currentSeason}}
            </div>
        `;

        stationList.innerHTML = 'Для этого города станции пока не добавлены.';
        return;
    }}

    if (!data) {{
        panel.innerHTML = `
            <div class="district-name">Нет данных</div>
            <div class="hint">Для этого района нет данных за ${{selectedYear}}, ${{currentSeason}}</div>
            <button class="top-btn" onclick="resetView()">Сбросить выбор</button>
        `;
        stationList.innerHTML = 'Нет станций';
        return;
    }}

    const avgValue = data.pollution ? data.pollution[currentPollutant] : null;

    panel.innerHTML = `
        <div class="district-name">${{data.district_ru}}</div>
        <div class="metric"><b>Город:</b> Алматы, Казахстан</div>
        <div class="metric"><b>Год:</b> ${{selectedYear}}</div>
        <div class="metric"><b>Сезон:</b> ${{currentSeason}}</div>
        <div class="metric"><b>Показатель:</b> ${{pollutantLabels[currentPollutant]}}</div>
        <div class="metric"><b>Среднее значение:</b> ${{formatValue(avgValue)}}</div>
        <div class="metric"><b>Количество измерений:</b> ${{data.count}}</div>
        <button class="top-btn" onclick="resetView()">Сбросить выбор</button>
    `;

    const stations = getStationsForDistrict(districtRu);

    if (stations.length === 0) {{
        stationList.innerHTML = 'Нет станций';
    }} else {{
        stationList.innerHTML = stations.map(s => `
            <div class="station-item">
                <b>${{s.station_name}}</b><br>
                ${{pollutantLabels[currentPollutant]}}: ${{formatValue(s.latest ? s.latest[currentPollutant] : null)}}<br>
                Последняя дата: ${{s.latest_date}}
            </div>
        `).join('');
    }}
}}

function resetDistrictStyles() {{
    for (let i = 0; i < allDistrictLayers.length; i++) {{
        allDistrictLayers[i].setStyle(getDefaultStyle());
    }}
    selectedLayer = null;
}}

function setAlmatyDistrictsVisible(visible) {{
    if (!geojsonLayer) return;

    if (visible) {{
        if (!map.hasLayer(geojsonLayer)) geojsonLayer.addTo(map);
    }} else {{
        if (map.hasLayer(geojsonLayer)) map.removeLayer(geojsonLayer);
    }}
}}

function resetView() {{
    selectedDistrictRu = null;
    resetDistrictStyles();
    clearMarkers();

    const cfg = cityConfig[currentCity];

    if (currentCity === "almaty") {{
        setAlmatyDistrictsVisible(true);
        pollutionLegendCard.style.display = "block";

        document.getElementById('info-card').innerHTML = `
            <div class="district-name">Алматы, Казахстан</div>
            <div class="hint">
                Год: ${{selectedYear || 'нет'}}<br>
                Сезон: ${{currentSeason || 'нет'}}<br>
                Показатель: ${{pollutantLabels[currentPollutant]}}<br>
                Нажми на район на карте.<br>
                Потом нажми на конкретную станцию.
            </div>
            <button class="top-btn" onclick="resetView()">Сбросить выбор</button>
        `;

        document.getElementById('station-list').innerHTML = 'Выбери район, потом станцию';
    }} else {{
        setAlmatyDistrictsVisible(false);
        pollutionLegendCard.style.display = "none";

        document.getElementById('info-card').innerHTML = `
            <div class="district-name">${{cfg.name}}</div>
            <div class="hint">
                Станций пока нет.<br>
                Сейчас доступен только каркас города и ArcGIS Land Cover слой.<br>
                Год: ${{selectedYear}}<br>
                Сезон: ${{currentSeason}}
            </div>
        `;

        document.getElementById('station-list').innerHTML = 'Для этого города станции пока не добавлены.';
    }}
}}

function changeCity() {{
    const cfg = cityConfig[currentCity];

    clearMarkers();
    resetDistrictStyles();
    map.setView(cfg.center, cfg.zoom);
    updateLandCover();
    resetView();
}}

geojsonLayer = L.geoJSON(geojsonData, {{
    style: function() {{
        return getDefaultStyle();
    }},
    onEachFeature: function(feature, layer) {{
        allDistrictLayers.push(layer);

        const districtRu = feature.properties?.district || feature.properties?.nameRu || feature.properties?.name || 'Unknown';
        const districtLabel = feature.properties?.nameRu || feature.properties?.district || feature.properties?.name || 'Unknown';

        layer.bindTooltip(districtLabel, {{
            sticky: true,
            direction: 'auto'
        }});

        layer.on('mouseover', function() {{
            if (currentCity !== "almaty") return;
            if (selectedLayer !== layer) layer.setStyle(getHoverStyle());
        }});

        layer.on('mouseout', function() {{
            if (currentCity !== "almaty") return;
            if (selectedLayer !== layer) layer.setStyle(getDefaultStyle());
        }});

        layer.on('click', function() {{
            if (currentCity !== "almaty") return;

            resetDistrictStyles();
            selectedLayer = layer;
            layer.setStyle(getHighlightStyle());

            updatePanel(districtRu);
            showMarkersForDistrict(districtRu);

            currentMarkersLayer.eachLayer(function(marker) {{
                if (marker.bringToFront) marker.bringToFront();
            }});

            if (typeof layer.getBounds === 'function') {{
                map.fitBounds(layer.getBounds(), {{ padding: [20, 20] }});
            }}
        }});
    }}
}}).addTo(map);

updateLandCover();
resetView();
</script>
</body>
</html>
"""

if os.path.exists(output_path):
    try:
        os.remove(output_path)
        print("old html removed")
    except Exception as e:
        print("could not remove old html:", e)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print("html saved")
print(output_path)
