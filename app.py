"""NIDS entry point. Start it from the project folder with:

    .\\.venv\\Scripts\\python.exe -m streamlit run app.py

This script only hands over to the shell, which draws the station stepper and the page the viewer opened. Nothing is
loaded or fitted here, so a rerun of this script is always cheap.
"""

from ui.shell import main

main()
