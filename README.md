# Agri-Doctor Pro

Leaf diagnosis for maize, rice and cotton (Streamlit).

1. A crop router suggests the crop. Maize and Rice suggestions always ask
   the user to confirm, because the router misroutes rice as maize (see
   "Known limitations").
2. The chosen crop's TFLite model reads the whole leaf (the verdict) and each
   quadrant (a check). Disagreement or low confidence is shown as
   "Needs review".
3. Field location comes from place search, the browser's location (on
   request only), or a click on the map; the field can be outlined by hand.
   Weather and modelled soil moisture come from the Open-Meteo API.
4. **Analyze All** runs the crop model, then asks Google Gemini for an
   independent look at the same photo (Gemini is not shown the model's
   result), and labels how the two relate: Agreement, Partial agreement,
   Disagreement or Uncertain. Both results are kept; confidences are never
   averaged. When they broadly agree, Gemini writes a cautious plan from
   both. If the key is missing, out of quota, rate-limited or offline, the
   ML diagnosis still works and the panel says why.

## Run locally

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Optional `.streamlit/secrets.toml` (git-ignored) for the Gemini features.
`GEMINI_API_KEY` (and optionally `GEMINI_MODEL`) environment variables are
used when there is no secrets file:

```toml
GEMINI_API_KEY = "your-key"
# optional, defaults to gemini-2.5-flash
gemini_model = "gemini-2.5-flash"
```

Models download from Google Drive on first use and are cached locally
(`*.tflite`, git-ignored). A download that is not a valid TFLite model is
rejected and deleted; it never replaces a working model.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest              # unit + real-model tests, no network
.venv/bin/python tests/e2e/run_e2e.py   # browser checks; live Drive/Open-Meteo/OSM
```

The browser run starts its own server from a copy without secrets, so it can
never call Gemini. Training pipeline smoke test: see `training/README.md`.

## Deploy on Streamlit Community Cloud

1. Push this folder to GitHub (see "Publishing safely").
2. At https://share.streamlit.io choose **Create app** → your repo, branch
   `main`, main file `app.py`.
3. **Advanced settings** → Python **3.12**. Under **Secrets** paste:
   `GEMINI_API_KEY = "your-key"` (only if you want AI features).
4. Deploy. First diagnosis downloads the models from Google Drive.

## Model files

| Crop | File | Drive ID | Outputs | Status |
|---|---|---|---|---|
| Router | crop_router_v2.tflite | 10LuRi-n6wFu4um9deAWBu0sr0-8e1MLV | 3 | works; rice/maize confusion |
| Maize | corn_model.tflite | 1_1PcQqUFFiK9tgpXwivM6J7OJShL18jk | 6 | works |
| Cotton | cotton_doctor_latest.tflite | 1tX19vWEBa9pyZCd_7hIGC8j2fwrHQEie | 10 | works; re-downloaded each diagnosis |
| Rice | rice_model.tflite (Drive name rice_weed_robust.tflite) | 1p2vZgq_FBigVnlhQPLQD4w2yjDn4zus3 | 8 | loads and runs; **accuracy not validated** (1 of 4 labelled test photos correct) |

All models take raw 0-255 RGB at 224x224; rescaling is inside each model.
Class order is in `agri_core.py`.

## Known limitations

- Crop router: on known photos (tests/data), 3 of 5 rice leaves were routed
  to Maize at 92-97% confidence. Retraining the router with more rice images
  is the real fix; the app asks the user to confirm instead.
- Quadrant crops lose leaf context and drift toward "Common Rust"; the whole
  leaf gives the verdict.
- Rice: the model's training code is not in this repository and its accuracy
  is unmeasured; every Rice result is marked "Needs review". Cotton's
  reported 95.6% test accuracy comes from `cotton_doctor_full.ipynb` (not
  re-measured here; no labelled cotton photos are in tests/data).
- Confidence scores are softmax outputs, not calibrated probabilities. No
  accuracy figures have been measured for any model in this repository.
- Field boundaries are drawn by the user; nothing is detected from imagery.
  Soil moisture is Open-Meteo's modelled estimate, not a measurement; soil
  type is user-entered.
- Saved fields live in the browser session; users download a JSON file to
  keep them. Nothing is stored on the server.
