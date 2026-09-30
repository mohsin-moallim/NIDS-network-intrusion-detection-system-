[![Github Profile][Jarvis1337]][Jarvis1337-url]
[![License][license-shield]][license-url]
[![Python Version][python]][python-url]
[![Streamlit][streamlit]][streamlit-url]
[![Scikit-learn][sklearn]][sklearn-url]
[![Version][version]][version-url]
[![Github Releases][github-releases]][github-releases-url]
[![Github Repo Size][GH-Repo]][GH-Repo-url]
[![Stars][stars]][stars-url]
[![Forks][forks]][forks-url]

*<h1 align="">🚀 Network Intrusion Detection System (NIDS) <3...</h1>*

*An intelligent Network Intrusion Detection System powered by advanced machine learning algorithms. This professional-grade application monitors network traffic in real-time, detects suspicious activity and known threats, and provides comprehensive analytics through an intuitive Streamlit interface. Built with multiple ML models including Random Forest, Gradient Boosting, SVM, and Neural Networks for superior threat detection accuracy.*

---

## *✨ Key Features*

*The Advanced AI NIDS comes packed with professional features:*

- ***🤖 Multi-Model Machine Learning:*** *Train and compare 4 powerful ML algorithms simultaneously (Random Forest, Gradient Boosting, SVM, Neural Network)*
- ***📊 Real-Time Traffic Analysis:*** *Live detection and classification of network traffic with instant threat alerts*
- ***📈 Advanced Analytics Dashboard:*** *Interactive visualizations including ROC curves, precision-recall curves, confusion matrices, and feature importance analysis*
- ***🎯 Multi-Model Consensus:*** *Get predictions from all models simultaneously for higher confidence in threat detection*
- ***📁 Flexible Data Input:*** *Support for CIC-IDS2017 dataset or built-in simulated network traffic data*
- ***📜 Training History & Reports:*** *Track model performance over time and export comprehensive analysis reports*
- ***🎨 Modern UI/UX:*** *Professional dark mode interface with responsive design and interactive components*
- ***⚡ High Performance:*** *Optimized for fast training and real-time predictions with parallel processing*

---

> [!IMPORTANT]  
> *Before you begin, ensure you have the following installed on your local machine:*
>
> - ***Python:*** *Version 3.8 or higher (Python 3.10+ recommended for best performance)*
> - ***Minimum 8GB RAM*** *for training ML models*
> - ***~1GB free disk space*** *for dependencies and datasets*
    
---

*<h2>📁 Project Structure</h2>*

*Your project directory should look like this:*

```
Network-Intrusion-Detection/
│
├── nids_app.py               # Main Streamlit application
├── requirements.txt          # Python dependencies
│
└── Datasets/                 # Place your CSV files here
    ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    ├── Monday-WorkingHours.pcap_ISCX.csv
    └── ... (other CIC-IDS2017 CSV files)
```

> [!NOTE]  
> *The `Datasets` folder is **already included** in the repository. You can add your own CIC-IDS2017 CSV files here, or use the built-in simulated data feature.*

---

*<h2>🏁 Installation Guide</h2>*

*Follow these steps to set up the project environment and install the necessary dependencies.*

### *1. Clone the Repository*

*First, clone the project repository from GitHub to your local machine and navigate into the project directory.*

```bash
git clone https://github.com/Jarvis1337/Network-Intrusion-Detection.git
cd Network-Intrusion-Detection
```

### *2. Download Dataset (Optional)*

*For real-world network traffic analysis, download the **CIC-IDS2017 dataset***:

- *Visit the [CIC-IDS2017 Dataset Page](https://www.unb.ca/cic/datasets/ids-2017.html)*
- *Download the CSV files for different days/attack types*
- *Extract and place all CSV files in the `Datasets/` folder*

> [!TIP]
> *If you don't have the dataset, the application will automatically generate simulated network traffic data for testing and demonstration purposes.*

### *3. Create a Virtual Environment*

*It is highly recommended to use a virtual environment to manage dependencies for this project. This keeps your project libraries isolated from your global Python installation.*

- *Run the following command to create a virtual environment named `venv`:*

```bash
python -m venv venv
```

### *4. Activate the Virtual Environment*

*Once the virtual environment is created, you need to activate it. The command differs depending on your operating system.*

- ***For Linux / macOS (Bash/Zsh):***

```bash
source venv/bin/activate
```

- ***For Windows (PowerShell):***

```powershell
# Optional: Run this if you get a permission error
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the environment
.\venv\Scripts\activate
```

- ***For Windows (Command Prompt):***

```cmd
.\venv\Scripts\activate.bat
```

> [!NOTE]  
> *Once activated, you should see `(venv)` appear at the beginning of your terminal prompt, indicating the virtual environment is active.*

### *5. Install Dependencies*

*With the virtual environment active, install all the required Python packages listed in the `requirements.txt` file using `pip`:*

```bash
pip install -r requirements.txt
```

*This will install the following packages:*
- *matplotlib==3.10.8 - For plotting and visualizations*
- *numpy==2.4.0 - For numerical computations*
- *pandas==2.3.3 - For data manipulation*
- *scikit-learn==1.8.0 - For machine learning algorithms*
- *seaborn==0.13.2 - For statistical visualizations*
- *streamlit==1.52.2 - For web interface*
- *plotly==6.5.0 - For interactive charts*

---

*<h2>🚀 Usage Guide</h2>*

### *Running the Application*

- *To start the Network Intrusion Detection System interface, use the `streamlit` command pointing to the main application file:*

```bash
streamlit run nids_app.py
```

### *Accessing the Interface*

- *After running the command, Streamlit will start a local server*
- *Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)*
- *The NIDS dashboard will load with a professional interface*

### *Using the Application*

*<h4>1. Configure Settings (Sidebar)</h4>*

- ***📁 Dataset Selection:*** *Choose a CSV file from your `Datasets` folder, or use simulated data*
- ***🤖 Model Selection:*** *Select one or more ML models to train (Random Forest, Gradient Boosting, SVM, Neural Network)*
- ***📊 Training Options:*** *Adjust simulated data size (1000-5000 samples)*

*<h4>2. Train Models</h4>*

- *Click the **"🚀 Train Models Now"** button in the sidebar*
- *Wait for models to train (progress bar will show training status)*
- *View results across multiple tabs*

*<h4>3. Explore Features</h4>*

- ***📊 Dataset Overview:*** *View dataset statistics, sample data, and attack distribution*
- ***🤖 Model Performance:*** *Compare accuracy, precision, recall, F1-score, and confusion matrices*
- ***📈 Advanced Analytics:*** *Analyze ROC curves, precision-recall curves, feature importance, and correlations*
- ***🔴 Live Detection:*** *Test real-time traffic detection with custom parameters or multi-model consensus*
- ***📜 History & Reports:*** *Track training history, export results to CSV, and view system statistics*

---

## *🎯 Model Performance*

*The system supports four powerful machine learning algorithms:*

| *Model* | *Typical Accuracy* | *Speed* | *Best For* |
|---------|-------------------|---------|------------|
| *Random Forest* | *95-98%* | *Fast* | *General-purpose detection* |
| *Gradient Boosting* | *96-99%* | *Medium* | *High accuracy requirements* |
| *SVM* | *93-96%* | *Slow* | *Binary classification* |
| *Neural Network* | *94-97%* | *Medium* | *Complex pattern recognition* |

---

## *📊 Supported Attack Types*

*The system can detect the following types of network intrusions:*

- ***DDoS*** *(Distributed Denial of Service)*
- ***DoS*** *(Denial of Service)*
- ***PortScan*** *(Port Scanning)*
- ***BruteForce*** *(Brute Force Attacks)*
- ***And more from CIC-IDS2017 dataset*

---

## *🛠️ Troubleshooting*

> [!CAUTION]
> ***Common Issues and Solutions:***
>
> - ***'streamlit' is not recognized:*** *Ensure you have activated your virtual environment before running the command. If the issue persists, try running:*
>   ```bash
>   python -m streamlit run nids_app.py
>   ```
>
> - ***Permission Denied on Windows:*** *If you cannot activate the virtual environment, run PowerShell as Administrator and execute:*
>   ```powershell
>   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
>   ```
>
> - ***Module Import Errors:*** *Make sure all dependencies are installed correctly:*
>   ```bash
>   pip install -r requirements.txt --upgrade
>   ```
>
> - ***No CSV files found:*** *Create the `Datasets` folder and add CSV files, or use the built-in simulated data feature*
>
> - ***Memory Errors during training:*** *Reduce the simulated data size in the sidebar (try 1000-2000 samples) or close other applications*

---

## *💡 Tips for Best Results*

- *Start with **Random Forest** and **Gradient Boosting** for optimal performance*
- *Use the **Multi-Model Consensus** feature for higher confidence predictions*
- *Train with real CIC-IDS2017 data for production-grade accuracy*
- *Regularly export training history to track model improvements*
- *Monitor the **Live Detection** tab for real-time threat analysis*

---

## *🤝 Contributing*

*Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Jarvis1337/Network-Intrusion-Detection/issues).*

---

## *📝 License*

> *<h4 align="center">⭐ If this project helped you, please consider giving it a star on GitHub!</h4>*

---

> *<h4 align="center">Network-Intrusion-Detection (NIDS) © 2026 by ~Jarvis is licensed under GNU General Public License v3.0 and Attribution 4.0 International</h4>*

<!-- Badge URLs -->
[Jarvis1337]: https://img.shields.io/badge/Github-Jarvis1337-blueviolet?style=for-the-badge&logo=github
[Jarvis1337-url]: https://github.com/Jarvis1337
[license-shield]: https://img.shields.io/github/license/Jarvis1337/Network-Intrusion-Detection?style=for-the-badge&logo=Github&color=E6E6FA
[license-url]: https://github.com/Jarvis1337/Network-Intrusion-Detection/blob/master/LICENSE
[python]: https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[streamlit]: https://img.shields.io/badge/Streamlit-1.52.2-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white
[streamlit-url]: https://streamlit.io/
[sklearn]: https://img.shields.io/badge/Scikit--Learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[sklearn-url]: https://scikit-learn.org/
[version]: https://img.shields.io/badge/Version-v3.1.0-brightgreen?style=for-the-badge&logo=Github&label=NIDS
[version-url]: https://github.com/Jarvis1337/Network-Intrusion-Detection/releases
[github-releases]: https://img.shields.io/badge/Github-Releases-ff0000?style=for-the-badge&logo=github
[github-releases-url]: https://github.com/Jarvis1337/Network-Intrusion-Detection/releases
[GH-Repo]: https://img.shields.io/github/repo-size/Jarvis1337/Network-Intrusion-Detection?style=for-the-badge&color=00ffff&label=Repository%20Size&logo=github
[GH-Repo-url]: https://github.com/Jarvis1337/Network-Intrusion-Detection/
[stars]: https://img.shields.io/github/stars/Jarvis1337/Network-Intrusion-Detection?style=for-the-badge&logo=github&color=yellow
[stars-url]: https://github.com/Jarvis1337/Network-Intrusion-Detection/stargazers
[forks]: https://img.shields.io/github/forks/Jarvis1337/Network-Intrusion-Detection?style=for-the-badge&logo=github&color=blue
[forks-url]: https://github.com/Jarvis1337/Network-Intrusion-Detection/network/members
