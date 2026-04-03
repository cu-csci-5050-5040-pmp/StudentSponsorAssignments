# Student-Project Assignment Optimizer

This is a Streamlit app for reviewing and improving student-to-project assignments against each student's top 4 preferences.

The app lets you:

- Upload a `preferences.csv` file and an `assignments.csv` file
- Adjust the penalty for non-preferred assignments
- View per-student assignment costs and total cost
- Edit assignments directly in the UI
- Run a greedy improvement pass
- Run an exact Hungarian optimization when the number of students matches the number of unique projects
- Visualize results with a cost distribution chart and a Sankey diagram
- Download the updated assignments with computed costs

## Files

- `app.py`: main Streamlit application
- `requirements.txt`: Python dependencies
- `.python-version`: pins Python for deployment platforms such as Render
- `preferences.csv`: sample/input preference data
- `assignments.csv`: sample/input assignment data

## Input Format

### `preferences.csv`

Required columns:

- `student`
- `pref1`
- `pref2`
- `pref3`
- `pref4`

Example:

```csv
student,pref1,pref2,pref3,pref4
Alice,GridIt!,Cisco #1,BioCollate,GriffMonster
Bob,Cisco #2,GridIt!,Credible,BioCollate
```

### `assignments.csv`

Required columns:

- `student`
- `project`

Example:

```csv
student,project
Alice,Cisco #1
Bob,GridIt!
```

## Run Locally

From the project root:

```powershell
cd "C:\Users\sindh\Documents\CUBoulder\CM_Fall2025\SProCo"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Deploy on Render

This app can be deployed as a Render Web Service.

### Render Settings

Use these settings in Render:

#### Build Command

```bash
pip install -r requirements.txt
```

#### Start Command

```bash
streamlit run app.py
```

## How to Use the App

1. Upload `preferences.csv` and `assignments.csv` in the sidebar, or enable the built-in example mode.
2. Set the non-preferred cost.
3. Click `Evaluate`.
4. Review the current assignment table and cost table.
5. Optionally edit assignments manually.
6. Optionally run `Greedy improve` or `Exact optimize`.
7. Download `assignments_with_cost.csv`.

## Notes

- Student and project names are normalized to reduce matching issues caused by whitespace or formatting differences.
- The app includes a small built-in example dataset for quick testing.
