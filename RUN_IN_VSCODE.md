# Run in VS Code (5 steps)

1. Install Python 3.10+ (python.org). On Windows tick "Add Python to PATH".
2. Unzip this folder, then in VS Code choose File > Open Folder and select `fraud_detection`.
3. Open a terminal: Terminal > New Terminal. Create an environment and install packages:

   Windows:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Mac / Linux:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   If VS Code asks to select an interpreter, pick the one inside `.venv`.
4. Train the model:
   ```
   python train.py
   ```
5. Start the app:
   ```
   streamlit run app.py
   ```
   Your browser opens at http://localhost:8501.

Optional: `python export_dashboard.py` creates `dashboard.html`. Double-click it to open the web dashboard.

You can also use the Run and Debug panel (left bar) and pick "1. Train model", "2. Run dashboard" or "3. Export web dashboard".

If `python` is not found, try `python3` or `py`. If `pip` fails, use `python -m pip install -r requirements.txt`.
