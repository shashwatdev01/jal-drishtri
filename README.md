# JAL-DRISHTI — SIH26015 Prototype

## What this is
A presentation-ready working prototype demonstrating the proposed JAL-DRISHTI workflow:
Geo-coded imagery → preprocessing → land-cover interpretation → GIS-style priority analysis → explainable recommendations → report.

## Important
This is a prototype, not a validated production watershed model. The demo uses transparent image-processing heuristics so the workflow works offline and reliably. For the SIH presentation, describe it as a functional prototype / proof of concept and state that the production version will use validated geospatial datasets and trained ML models.

## Run
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:
   `pip install -r requirements.txt`
4. Start:
   `streamlit run app.py`
5. The browser will open automatically.

## Demo
Keep "Use built-in demo landscape" ON for the safest presentation.
Then show:
1. Input imagery
2. Segmentation
3. Priority heatmap
4. Priority zones
5. Explainable recommendations
6. Architecture
7. Downloadable report

## Presentation wording
Say:
"Today's prototype demonstrates the complete decision-support pipeline. The segmentation and priority scoring shown here are proof-of-concept components; in the next stage, they will be replaced or calibrated with validated ML models, authoritative geospatial datasets and field verification."

Do NOT claim the demo's scores are field-validated accuracy numbers.
