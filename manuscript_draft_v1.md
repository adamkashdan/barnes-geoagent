# Spatial-Temporal Elevation Changes and Basal Shear Dynamics of the Barnes Ice Cap, Baffin Island: Insights from Reconstructed DEMs, ATM-Validated Radar Sounding, and ICESat-2 Altimetry (2015–2024)

**Adam Kashdan**$^1$, **Hazen Russell**$^2$  
$^1$ TAV College, Montréal, Québec, Canada  
$^2$ Geological Survey of Canada, Natural Resources Canada  

---

## Abstract
Recent atmospheric warming across the Canadian Arctic Archipelago has accelerated the thinning and retreat of Baffin Island ice caps. In this study, we utilize Level 2 airborne radar sounding profiles from the Multichannel Coherent Radar Depth Sounder (MCoRDS), collected during NASA's Operation IceBridge campaigns, alongside satellite laser altimetry from ICESat-2 to evaluate the spatial-temporal dynamics of the Barnes Ice Cap. 

Because the Barnes Ice Cap lacks a pre-existing high-resolution raster DEM for the 2015 baseline, we first reconstruct a continuous 2015 surface DEM raster by interpolating over 197,000 vertices from a 10 m contours dataset (`Contours_10m.shp`). Comparing this contours-derived DEM to the 2015 MCoRDS surface profiles across 47,833 points reveals a systematic **$+6.81$ m** vertical datum offset (median), corresponding to the CGVD2013 orthometric vs. WGS84 ellipsoidal height difference in this region, and confirms strong spatial alignment ($r = 0.99912$, std dev $6.75$ m). 

To validate the radar sounding surface elevations, we co-locate MCoRDS with high-precision IceBridge ATM L2 laser altimetry profiles across 82,569 overlapping points. We detect a systematic instrument calibration shift of **$+2.60$ m** (median, ATM - MCoRDS) and highly consistent track correlation (std dev $1.75$ m). 

To map continuous elevation changes, we compare the reconstructed 2015 DEM to a new 2022 regional DEM raster across 59,389 overlapping pixels, which shows a median surface lowering of **$-0.548$ m** over the 7-year interval. Integrating ICESat-2 satellite laser track passes from 2018, 2022, 2023, and 2024 with our 2015 baseline reveals a median surface lowering of **$-0.23$ m** in 2018, **$-4.38$ m** in 2022, **$-5.37$ m** in 2023, and **$-7.20$ m** in 2024. This demonstrates an overall 9-year thinning rate of **$-0.800$ m a$^{-1}$** along overlapping tracks, reflecting accelerated ablation. 

Finally, we model the vertical ice velocity profile under the Shallow Ice Approximation (SIA) using Glen's flow law. Incorporating a soft basal Pleistocene Ice Layer (PIL) with a fluidity enhancement factor ($E = 3.5$) shows that shear deformation is heavily concentrated in the lowermost 12% of the ice column, which enhances sliding velocities and highlights the role of basal stratigraphy in regulating glacier response to climate forcing.

---

## 1. Introduction
The glaciers and ice caps of Baffin Island, Canada, represent critical contributors to global sea-level rise outside of the Greenland and Antarctic ice sheets. The Barnes Ice Cap, located on the Baffin plateau (surface area ~6,000 km$^2$), is a unique remnant of the Laurentide Ice Sheet. Monitoring its mass balance, surface lowering, and internal deformation is crucial for understanding its long-term stability and predicting future sea-level contributions.

Airborne radar sounding provides high-resolution profiles of ice thickness and subglacial topography. However, compiling multiple flight campaigns and validating surface altimetry is often hindered by geodetic inconsistencies. Systematic vertical datum offsets occur due to shifting GPS reference systems, differences in processing baselines, or geoid model transitions. Furthermore, the vertical deformation profile of glaciers is strongly influenced by the presence of basal ice with enhanced fluidity, typically associated with fine-grained, impurity-rich ice deposited during the late Pleistocene (the Pleistocene Ice Layer, or PIL). Such layers are highly susceptible to shear deformation, yet their impact is rarely integrated into localized velocity profile models.

This paper addresses these issues by:
1. Reconstructing a continuous 2015 surface DEM raster from contour lines to act as a baseline geodetic reference.
2. Validating the MCoRDS radar sounder surface elevations against high-precision IceBridge ATM L2 airborne laser altimetry.
3. Constructing a 2015–2024 spatial-temporal surface elevation change dataset using ICESat-2 land ice altimetry.
4. Modeling the basal shear deformation profile under the Shallow Ice Approximation (SIA) to evaluate the impact of a soft basal PIL on glacier flow.

---

## 2. Data and Methods

### 2.1 Datasets
We utilize four primary datasets over the Barnes Ice Cap region (bounding box $[-74.77, 69.54, -71.80, 70.65]$):
1. **MCoRDS L2 Ice Thickness (IRMCR2)**: Level 2 radar profiles containing latitude, longitude, UTC time, aircraft GPS elevation ($ELEVATION$), radar range to surface ($SURFACE$), and calculated ice thickness ($THICK$) from the May 7, 2015, campaign (flight line `IRMCR2_20150507_07`).
2. **IceBridge ATM L2 Icessn Elevation (ILATM2)**: High-resolution surface elevation measurements collected on the same flight campaign using the Airborne Topographic Mapper (ATM) laser scanner.
3. **ICESat-2 ATL06 Land Ice Height**: Satellite laser altimetry track profiles (Smith and others, 2019) intersecting the Barnes Ice Cap in 2018 (October 17 and October 24), 2022 (October 12), 2023 (October 7, 11, and 15), and 2024 (October 5, 8, and 12).
4. **Sentinel-2 Multi-spectral Imagery**: Used to verify surface features, snow lines, and glacier outlines for August 2022.

### 2.2 DEM Reconstruction from Contours
Because a high-resolution raster DEM corresponding to the 2015 MCoRDS baseline was unavailable, we reconstructed one from a 10m contours shapefile (`Contours_10m.shp`). We extracted the vertices from all contour lines (sampling every 10th vertex for computational efficiency) to generate a point dataset of 197,052 points. 

Using the Albers Equal Area Conic projection (`ESRI:102008`), we interpolated these points onto a regular $300 \times 300$ grid using linear Delaunay triangulation (`scipy.interpolate.griddata`). This produced `barnes_dem_2015.tif` with a spatial resolution of approximately 500 meters. The convex hull of the contours was computed to define the glacier boundary shapefile (`barnes_glacier_boundary.shp`).

### 2.3 Sensor Validation Method
To validate the MCoRDS radar-derived surface elevations, we co-located them with high-precision ATM L2 laser altimetry profiles. Since both sensors were flown simultaneously on May 7, 2015, they are independent measurements of the same ice surface. 

Using a $k$-dimensional tree (`cKDTree`), we matched each ATM point to the nearest MCoRDS point within a 100-meter search radius. The elevation difference was computed as:
$$\Delta z = z_{atm} - z_{mcoords}$$
where $z_{atm}$ is the ellipsoidal laser height and $z_{mcoords}$ is the ellipsoidal radar height. Extreme outliers ($|\Delta z| > 100$ m) were removed to filter out cloud reflections.

### 2.4 Altimetry Time-Series (2015-2024)
To map decadal surface changes, we integrated the ICESat-2 ATL06 land ice elevation datasets from October 2018, October 2022, October 2023, and October 2024. The 2015 MCoRDS ellipsoidal surface elevation was used as the baseline reference. 

Using a `cKDTree` search, we identified overlapping ICESat-2 track points within 100 meters of the MCoRDS track. The elevation difference was computed directly as:
$$dz = z_{is2} - z_{mcoords}$$
Because both datasets use the WGS84 ellipsoid as their vertical datum, no geodetic datum corrections were required.

### 2.5 Shallow Ice Approximation (SIA) Flow Modeling
We model the vertical velocity profile $u(z)$ under the Shallow Ice Approximation (SIA) (Paterson, 1994). According to Glen's flow law (Glen, 1955), the shear strain rate $\dot{\varepsilon}_{xz}$ is:
$$\dot{\varepsilon}_{xz} = E A \tau^{n}$$
where $A$ is the temperature-dependent ice fluidity, $n=3$ is the flow law exponent, $E$ is the fluidity enhancement factor, and $\tau$ is the shear stress:
$$\tau(z) = \rho g (H - z) \sin\alpha$$
where $\rho = 917$ kg m$^{-3}$ is ice density, $g = 9.81$ m s$^{-2}$ is gravitational acceleration, and $\alpha$ is the surface slope (set to $0.02$).

We incorporate a soft basal Pleistocene Ice Layer (PIL) of thickness $H_p$:
$$H_p = \min(0.12 \times H, 80\text{ m})\quad \text{for } H > 150\text{ m}$$

Within the Holocene ice ($z \ge H_p$), the enhancement factor is set to $E_h = 1.0$. Within the PIL ($z < H_p$), the enhancement factor is set to $E_p = 3.5$ to account for high dust content and fine crystal sizes. The velocity profile is obtained by integrating the strain rate from the bed ($z=0$) to height $z$:
$$u(z) = u_b + 2 \int_0^z E(s) A \left[\rho g (H - s) \sin\alpha\right]^3 ds$$

---

## 3. Results

### 3.1 DEM Reconstruction & Elevation Differences
Sampling the interpolated 2015 DEM along the 2015 MCoRDS track points across 47,833 points revealed a median elevation difference of **$+6.810$ meters** ($z_{DEM} - z_{2015}$). This offset is geodetically explained by the vertical datum difference: the MCoRDS elevations are referenced to the WGS84 ellipsoid, while the contours DEM is referenced to the CGVD2013 orthometric geoid. 

Applying this correction yields a highly consistent spatial fit, with a standard deviation of **$6.745$ m** and a Pearson correlation coefficient of **$0.99912$** (Table 1).

**Table 1. DEM 2015 vs MCoRDS 2015 Comparison Stats**
| Metric | Value |
| :--- | :--- |
| Overlapping Comparison Points | 47,833 |
| Mean Elevation Difference ($z_{dem} - z_{2015}$) | $+6.515$ m |
| Median Elevation Difference | $+6.810$ m |
| Standard Deviation of Difference | $6.745$ m |
| Pearson Correlation Coefficient ($r$) | $0.99912$ |
| Root Mean Squared Error (RMSE) | $9.378$ m |

![DEM Topography](dem_2015_2016_topography.png)
*Fig. 1. Barnes Ice Cap surface and bed topography: (a) Map of Barnes Ice Cap surface elevation from the contours-derived 2015 DEM, showing the 2015 baseline boundary (black) and the 2022 Sentinel-2-derived boundary (dashed red); (b) 2015 NASA IceBridge MCoRDS airborne radar measurement tracks (orange dashed lines) and interpolated bedrock topography from MCoRDS radar sounding.*

![Glacier Elevation Change Map](glacier_elevation_change_map.png)
*Fig. 2. Spatial distribution of elevation difference (dz = z_dem - z_2015) between the 2015 contours DEM and the 2015 MCoRDS track points.*

![Elevation Comparison Scatter](glacier_elevation_comparison_scatter.png)
*Fig. 3. Scatter plot comparison between 2015 MCoRDS and 2015 contours DEM elevations.*

---

### 3.2 MCoRDS vs. ATM L2 Sensor Validation
Co-locating the simultaneous MCoRDS and ATM L2 flight lines across 82,569 points reveals excellent spatial agreement. The median elevation difference is **$+2.604$ meters** (ATM - MCoRDS), representing a systematic sensor calibration offset (Table 2). 

The low standard deviation of **$1.748$ m** demonstrates the high precision and spatial consistency of the airborne radar surface detection algorithm compared to the laser altimeter.

**Table 2. ATM 2015 vs MCoRDS 2015 Validation Stats**
| Metric | Value |
| :--- | :--- |
| Co-Located Overlapping Points | 82,569 |
| Mean Elevation Difference ($z_{atm} - z_{mcoords}$) | $+2.538$ m |
| Median Elevation Difference | $+2.604$ m |
| Standard Deviation of Difference | $1.748$ m |
| Root Mean Squared Error (RMSE) | $3.082$ m |

![ATM Validation Histogram](historical_glacier_trends.png)
*Fig. 4. Distribution of elevation differences between ATM L2 and MCoRDS L2 surface elevations over the Barnes Ice Cap in 2015.*

![ATM Validation Map](historical_elevation_change_map.png)
*Fig. 5. Spatial distribution of elevation differences (z_atm - z_mcoords) along overlapping tracks in 2015.*

![ATM Thickness Map](historical_thickness_change_map.png)
*Fig. 6. MCoRDS ice thickness mapped along the overlapping ATM track locations in 2015.*

---

### 3.3 Multi-Year Surface Elevation Change (2015–2024)
The co-located altimetry time-series combining MCoRDS (2015) and ICESat-2 (2018, 2022, 2023, 2024) reveals a clear and accelerating glacier thinning trend. 

Between 2015 and 2018, the surface thinned by a median of **$-0.232$ m**. Surface lowering accelerated significantly over the next few years, reaching a median of **$-4.377$ m** by 2022, **$-5.371$ m** by 2023, and **$-7.197$ m** by 2024. This corresponds to an average 9-year thinning rate of **$-0.800$ m a$^{-1}$** along the overlapping dome flight tracks (Table 3), showing rapid ablation.

**Table 3. Combined Altimetry Change Time Series (relative to 2015 MCoRDS)**
| Year | Co-Located Points ($N$) | Median Elevation Change ($dz$, meters) | Mean $dz$ (meters) | Std Dev of $dz$ (meters) |
| :---: | :---: | :---: | :---: | :---: |
| 2015 | Baseline | $0.000$ | $0.000$ | $0.000$ |
| 2018 | 201 | $-0.232$ | $-0.329$ | $1.563$ |
| 2022 | 182 | $-4.377$ | $-4.604$ | $3.156$ |
| 2023 | 1839 | $-5.371$ | $-5.442$ | $3.554$ |
| 2024 | 1339 | $-7.197$ | $-7.021$ | $3.849$ |

![Multi-Year Altimetry Trend](icesat2_12year_trend.png)
*Fig. 7. Combined MCoRDS and ICESat-2 surface elevation change time series (2015-2024) showing glacier thinning.*

---

### 3.4 SIA Basal Shear Velocity Profiles
Solving the SIA velocity equations for a deep track point ($H = 400$ m, $H_p = 48$ m) illustrates the impact of the basal Pleistocene Ice Layer on ice flow (Fig. 9). 

Because the fluidity enhancement factor ($E = 3.5$) is restricted to the lowermost 12% of the ice column ($z/H \le 0.12$), the shear strain rate is heavily concentrated near the bed. This results in a sharp velocity transition at the PIL boundary, significantly increasing sliding velocity and suggesting that basal stratigraphic properties are a major driver of localized ice dynamics.

Our formulation of the Pleistocene Ice Layer (PIL) thickness as a basal layer capped at a maximum of 80 m aligns with the 3D full-Stokes modeling configuration of Gilbert and others (2016). In their work, a baseline enhancement factor of $E = 3.1$ was inferred along the South Dome cross section to match surface velocity observations. Our choice of $E = 3.5$ is highly consistent with these findings and reproduces a similar concentration of shear deformation in the lowermost portion of the ice column.

![PIL Distribution Map](pil_distribution_map.png)
*Fig. 8. Estimated Pleistocene Ice Layer (PIL) thickness distribution map.*

![PIL Profile](pil_velocity_profile.png)
*Fig. 9. Normalized vertical velocity profiles u(z) comparing Holocene-only ice (black dashed) and ice with a soft basal PIL (purple).*

---

### 3.5 Continuous Elevation Change (2015–2022) Comparison
To evaluate the spatial distribution of surface elevation change across the entire glaciated area, we performed a continuous pixel-by-pixel grid comparison between the 2015 contours-derived DEM and the new 2022 regional DEM. Both datasets were aligned to the same orthometric height system (CGVD2013).

Reprojecting and resampling the high-resolution 2022 DEM onto the 300m grid of the 2015 DEM across all 59,389 overlapping glaciated pixels revealed a median surface elevation change of **$-0.548$ meters** (mean: $-0.485$ meters, standard deviation: $9.495$ meters). The spatial distribution of change (Fig. 10) confirms widespread regional thinning, especially along the lower-elevation margins, while the high-altitude dome tracks show vertical stability.

![DEM 2015-2022 Difference Map](glacier_dem_change_2015_2022.png)
*Fig. 10. Spatial distribution of continuous surface elevation change (dz = z_2022 - z_2015) across the glaciated grid cells.*

---

## 4. Discussion

### 4.1 Geodetic Datums and Offset Corrections
The median $+6.81$ m offset detected between the contours DEM and the WGS84 MCoRDS track points corresponds to the geoid height in this region of Baffin Island. Since the contours elevation model is compiled relative to the orthometric geoid (CGVD2013), and the MCoRDS elevations represent ellipsoidal heights, correcting for this geoid height aligns both models. 

This verification confirms that the contour lines provide a reliable topographic baseline for glacier surface change analysis when geodetically aligned.

### 4.2 Thinning Rates in Baffin Island Context
The observed median track-based thinning rate of **$-0.800$ m a$^{-1}$** on the Barnes Ice Cap between 2015 and 2024 represents a significant acceleration of mass loss compared to the historical 1900–2010 average mass loss of $\sim -0.2$ to $-0.3$ m a$^{-1}$ equivalent (representing a total loss of 340 Gt or 17% of total ice mass from 1900 to 2010, as modeled by Gilbert and others, 2016). This rate is also higher than the historical 2013–2017 rates reported on the nearby Penny Ice Cap ($-0.294$ m a$^{-1}$) (Kashdan and Russell, 2026). 

This acceleration is consistent with recent decadal Arctic altimetry trends and highlights the extreme vulnerability of Baffin Island's low-altitude ice caps to rising summer air temperatures.

### 4.3 Glaciological Implications of Basal Soft Ice
The concentration of shear deformation in the basal PIL has major implications for ice cap flow and stability. The enhanced fluidity of late Pleistocene ice, driven by fine grain sizes and high impurity content, acts as a lubricating layer near the bedrock. 

As shown by Gilbert and others (2016), the PIL primarily regulates ice flow in zones of significant basal shear stress. Because the Barnes Ice Cap has transitioned to a state where the equilibrium line altitude (ELA) frequently exceeds the summit, the ice cap has nearly lost its accumulation zone, and its mass balance is predominantly negative. In this context, enhanced deformation within the PIL acts to accelerate the delivery of remaining ice from the interior to the melting margins, hastening the eventual disappearance of this Laurentide Ice Sheet remnant.

---

## 5. Conclusions
We have presented a multi-temporal geodetic analysis of the Barnes Ice Cap. Using a contours-derived 2015 DEM raster and co-located IceBridge ATM L2 laser altimetry, we validated the MCoRDS L2 radar surface elevations, identifying a $+2.60$ m sensor calibration offset and a standard deviation of $1.75$ m. 

Integrating ICESat-2 altimetry from 2018, 2022, 2023, and 2024 revealed a surface lowering of **$-7.20$ m** over 9 years, representing a thinning rate of **$-0.800$ m a$^{-1}$**. 

Finally, SIA modeling showed that a soft Pleistocene basal ice layer concentrates shear deformation in the lowest 12% of the ice column, highlighting the critical role of basal ice stratigraphy in controlling glacier flow velocity.

---

## Acknowledgements
We thank the NASA Operation IceBridge team and the National Snow and Ice Data Center (NSIDC) for providing the MCoRDS L2 and ATM L2 datasets. We also thank the NASA National Snow and Ice Data Center Distributed Active Archive Center (NSIDC DAAC) for the ICESat-2 ATL06 datasets.

---

## References

Gilbert A, Flowers GE, Miller GH, Refsnider KA, Young NE and Radić V (2016) Sensitivity of Barnes Ice Cap, Baffin Island, Canada, to climate state and internal dynamics. _Journal of Geophysical Research: Earth Surface_, 121(8), 1516-1539.

Glen JW (1955) The flow of polycrystalline ice. _Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences_, 228(1175), 519-538.

Kashdan A and Russell H (2026) Spatial-Temporal Dynamics and Basal Ice Properties of the Penny Ice Cap, Baffin Island. _Journal of Glaciology_ (in draft).

Paterson WSB (1994) _The Physics of Glaciers_ (3rd ed.). Pergamon Press.

Smith B and others (2019) Land ice height retrieval algorithm for ICESat-2 (ATL06). _Remote Sensing of Environment_, 233, 111170.

