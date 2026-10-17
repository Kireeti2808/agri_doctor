# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Primary: evaluators of a project demo (course reviewers, hackathon judges, faculty) who open the hosted app, upload a leaf photo, and judge the system within a few minutes. First impression and legibility of what the AI did matter most. The underlying use case it demonstrates is farm-level crop diagnosis.

## Product Purpose
Agri-Doctor Pro diagnoses crop leaf images. A router model identifies the crop (Maize, Rice, Cotton), a crop-specific model classifies diseases and weeds in each image quadrant, local weather is fetched for the farm location, and an AI agronomist (OpenAI) writes a management plan. Success: an evaluator understands each step of the pipeline and trusts the result enough to read the prescription.

## Positioning
A two-stage on-device pipeline (crop router, then crop-specific TFLite model) plus quadrant-level spatial analysis plus weather-aware spraying advice, in one flow.

## Operating Context
Hosted on Streamlit Community Cloud so it is reachable when the developer's Mac is off. Visitors use desktop browsers mostly (demo on laptops and projectors), sometimes phones.

## Capabilities and Constraints
- Streamlit app (`app.py`); styling is limited to what Streamlit allows (injected CSS, HTML blocks, native widgets).
- Models download from Google Drive via gdown; Router/Maize/Rice are cached, Cotton is re-downloaded on every diagnosis.
- Router confidence under 75% falls back to manual crop selection.
- Classes: Maize 6, Rice 8, Cotton 10, including weed classes.
- Weather: Open-Meteo geocoding and forecast (temperature, humidity, precipitation, WMO code).
- AI agronomist: OpenAI chat completions, key read from Streamlit secrets `openai_key`; toggleable.
- Output is an AI classification, not a confirmed laboratory diagnosis; the UI must say so.

## Evidence on Hand
No testimonials, accuracy benchmarks, or user counts exist. Do not invent model accuracy figures or customer claims.

## Product Principles
1. Show the pipeline: every stage (crop detection, quadrant scan, weather, prescription) is visible and legible.
2. Honest confidence: scores and the AI-not-lab caveat are always shown.
3. Demo-ready in one minute: upload to result with no setup.
