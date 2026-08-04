# Barnes Ice Cap GeoAgent 2015

An AI-powered geospatial assistant built on the Gemini API to analyze radar measurements of ice thickness, subglacial bedrock topography, and elevation change for the Barnes Ice Cap (Baffin Island, Nunavut, Canada). The dataset consists of real measurements collected on May 7, 2015, during the NASA Operation IceBridge survey using the Multichannel Coherent Radar Depth Sounder (MCoRDS L2) and the Airborne Topographic Mapper (ATM L2).

This project adapts the semantic layer and tool-use architecture of [penny-geoagent](https://github.com/adamkashdan/penny-geoagent) to work directly with Barnes Ice Cap flight lines, incorporating high-precision laser altimetry (ATM L2), satellite altimetry (ICESat-2 ATL06), and surface contour lines.

---

## Architecture

```
User question (natural language)
       │
       ▼
FastAPI interface /ask endpoint (src/main.py)
       │
       ▼
Agent loop (src/agent.py) <──> Gemini API (Function Calling)
       │
       ▼
GIS tools (src/tools.py) <──> Pandas / Matplotlib analysis over CSV
       │
       ▼
Semantic layer (semantic_layer.yaml) <── Describes dataset schema to the LLM
```

---

## Data Setup

Raw scientific datasets are **not included** in this repository due to file size limits and NASA data distribution policies. You will need to obtain the datasets and place them under the `data/` directory.

### Directory Structure
Create a `data/` folder in the project root and place the files as follows:
```
Barnes_2015/
├── data/
│   ├── 2015_85564591/
│   │   ├── Contours_10m.shp
│   │   └── ...
│   ├── 2015_85564591_v2/
│   │   ├── IRMCR2_20150507_07.csv
│   │   └── ...
│   ├── ICESat-2/
│   │   ├── elev_2018-10-17_t286_1681085410510/
│   │   └── ...
│   ├── IceBridge ATM L2 Icessn Elevation, Slope, and Roughness V002/
│   │   ├── 71410808/
│   │   └── ...
│   └── S2B_MSIL2A_20220807T170849_N0400_R112_T18WWC_20220807T211146.SAFE/
```
*(Note: `.gitignore` is configured to ignore the actual dataset files so they will not be committed to Git).*

---

## Installation & Setup

### 1. Create and Activate the Virtual Environment
Create and activate the virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install all required GIS and web packages (`pandas`, `geopandas`, `shapely`, `rasterio`, `matplotlib`, `google-genai`, `fastapi`, `uvicorn`, `scipy`):
```bash
pip install -r requirements.txt
```

### 3. Set Up Your Gemini API Key
Create a `.env` file in the root of the project and define your API key:
```env
GEMINI_API_KEY="your-api-key-here"
```

### 4. Verify GIS Tools
You can test the GIS query logic, stats computation, and map generation directly without calling the LLM:
```bash
python verify_tools.py
```
This will generate a test map named `test_pil_map.png` in the root folder.

### 5. Run the Agent in CLI Mode
You can ask the agent questions directly from the command line:
```bash
python src/agent.py "What is the average ice thickness on the Barnes Ice Cap in the bounding box [-73.5, 69.8, -73.0, 70.3]?"
```

### 6. Start the FastAPI Service
Launch the development server:
```bash
uvicorn src.main:app --reload --port 8000
```
Open your browser and navigate to `http://localhost:8000/docs` to view the interactive API documentation.

---

## Science and Modeling Modules

### 1. DEM Reconstruction from Contour Lines
Because a high-resolution raster DEM corresponding to the 2015 MCoRDS baseline was unavailable, the project includes an interpolation script:
* **Run command**: `python src/create_dem_from_contours.py`
* **Output**:
  1. `data/DEM_2015/barnes_dem_2015.tif` (Interpolated 2015 DEM raster)
  2. `data/DEM_2015/barnes_glacier_boundary.shp` (Glacier outline shapefile)

### 2. DEM vs. MCoRDS Comparison
Compares the interpolated 2015 DEM with the 2015 MCoRDS surface elevations:
* **Run command**: `python src/dem_analysis.py`
* **Geodetic Correction**: Identifies a systematic **$+6.810$ meters** vertical datum offset (median), reflecting the CGVD2013 orthometric vs. WGS84 ellipsoidal height difference in this region. Standard deviation after correction is $6.75$ m ($r = 0.99912$).

### 3. Sensor Validation (ATM vs. MCoRDS)
Compares MCoRDS radar-derived surface elevations with high-precision airborne laser (ATM L2) surface elevations from the same flight:
* **Run command**: `python src/historical_analysis.py`
* **Outputs**: Computes a systematic instrument calibration offset of **$+2.604$ meters** (median, ATM - MCoRDS) with a standard deviation of only **$1.748$ meters**, validating the radar surface alignment.

### 4. Altimetry Integration (2015-2022)
Combines MCoRDS (2015) and ICESat-2 (2018, 2022) to trace multi-temporal change:
* **Run command**: `python src/analyze_icesat2.py`
* **Outputs**: Quantifies a 7-year glacier surface lowering of **$-4.38$ meters** by 2022. This demonstrates an overall decadal thinning rate of **$-0.625$ m/year** along the overlapping tracks.

### 5. Pleistocene Ice Layer (PIL) Flow Modeling
Models the vertical velocity profile $u(z)$ under the Shallow Ice Approximation (SIA) using Glen's flow law:
* **Run command**: `python src/pil_modeling.py`
* **Outputs**: Incorporates a soft basal Pleistocene layer ($E=3.5$), showing that shear deformation is heavily concentrated in the lowermost 12% of the ice column.

---

## Example Questions to Ask the Agent:
* *"What is the average ice thickness on the Barnes Ice Cap?"*
* *"Show me a map of the ice thickness for the entire Barnes Ice Cap survey area."*
* *"Calculate the zonal statistics of the surface elevation in the bounding box [-73.5, 69.8, -73.0, 70.3]."*
* *"Is there a correlation between surface elevation and ice thickness in the central part of the ice cap?"*
* *"Generate a satellite NDSI map for August 7, 2022."*

---

## License & Copyright

Copyright (c) 2026 Adam Kashdan. All rights reserved.

This repository contains draft source code associated with an upcoming scientific publication. The code is provided solely for reference and academic peer-review purposes.

See the [LICENSE](LICENSE) file for full copyright terms and restrictions.

