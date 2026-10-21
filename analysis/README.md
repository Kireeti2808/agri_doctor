# Independent re-analysis and figure scripts

Written separately from the notebooks' helper modules; they recompute the paper's statistics from the per-image records.

- `reanalysis.py` / `figures.py`: first evaluation run. `python3 reanalysis.py <agri_results_v2> out.json`, `python3 figures.py <agri_results_v2> fig`
- `reanalysis_followup.py` / `figures_followup.py`: field adaptation run. `python3 reanalysis_followup.py <agri_results_followup> out.json`, `python3 figures_followup.py <agri_results_followup> <agri_results_v2> fig`

Unzip `agri_results_v2.zip` and `agri_results_followup.zip` from the repository root first.
