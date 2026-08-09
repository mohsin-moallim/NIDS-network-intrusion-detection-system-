import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from datetime import datetime
import io

st.set_page_config(
    page_title="Advanced AI Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

theme_enable = False

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 90px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
# st.markdown('<p class="main-header" style="font-size: 60px !important; line-height: 1.2;">🛡️ Advanced AI Network Intrusion Detection System</p>', unsafe_allow_html=True)

st.markdown("""
    <h1 class="main-header" style='text-align: center; font-size: 90px !important; color: #00d4ff; margin-bottom: 0px;'>
        🛡️ Advanced AI Network Intrusion Detection System
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; font-size: 20px; margin-bottom: 20px;'>
    <b>Professional Network Security Monitoring with Advanced Machine Learning</b><br>
    Multi-Model Analysis | Real-time Detection | Comprehensive Analytics
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'training_history' not in st.session_state:
    st.session_state['training_history'] = []
if 'prediction_count' not in st.session_state:
    st.session_state['prediction_count'] = {'normal': 0, 'attack': 0}
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'Dark Mode'
for _sim_key, _sim_default in {
    'sim_running': False,
    'sim_tick': 0,
    'sim_total': 0,
    'sim_pred_normal': 0,
    'sim_pred_attack': 0,
    'sim_correct': 0,
    'sim_feed': [],
    'sim_timeline': [],
    'sim_alerts': [],
    'sim_attack_type_counts': {},
}.items():
    if _sim_key not in st.session_state:
        st.session_state[_sim_key] = _sim_default

st.sidebar.header("⚙️ Advanced Control Panel")
st.sidebar.markdown("---")

# Theme Enable Logic
if theme_enable:
    available_themes = ["Dark Mode", "Light Mode"]

    current_val = st.session_state.get('theme', 'Dark Mode')
    if current_val not in available_themes:
        current_val = 'Dark Mode'
        
    theme = st.sidebar.selectbox("🎨 Theme", available_themes, 
    index=available_themes.index(current_val))
else:
    theme = 'Dark Mode'

# Update theme in session state
if theme != st.session_state['theme']:
    st.session_state['theme'] = theme
    st.rerun()

# Apply theme-specific CSS
if theme == "Dark Mode":
    st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #00d4ff;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }
        .stMetric {
            background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #00d4ff;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            box-shadow: 0 0 15px rgba(118, 75, 162, 0.8);
        }
    </style>
    """, unsafe_allow_html=True)
elif theme == "Light Mode":
    st.markdown("""
    <style>
        .stApp {
            background-color: #f8f9fa;
            color: #212529;
        }
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #0066cc;
            text-align: center;
            margin-bottom: 1rem;
        }
        .stMetric {
            background: linear-gradient(135deg, #ffffff 0%, #e9ecef 100%);
            padding: 1rem;
            border-radius: 10px;
            border: 2px solid #0066cc;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background: linear-gradient(135deg, #0066cc 0%, #004499 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #004499 0%, #0066cc 100%);
            box-shadow: 0 4px 8px rgba(0, 102, 204, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)

# Function to generate simulated data
def generate_simulation_data(num_samples=1000):
    """Generate realistic network traffic data for training"""
    np.random.seed(42)
    
    # Normal Traffic (70%)
    normal_samples = int(num_samples * 0.7)
    normal_data = {
        'Flow Duration': np.random.normal(2000000, 500000, normal_samples),
        'Total Fwd Packets': np.random.randint(1, 50, normal_samples),
        'Total Backward Packets': np.random.randint(1, 50, normal_samples),
        'Total Length of Fwd Packets': np.random.normal(1000, 300, normal_samples),
        'Total Length of Bwd Packets': np.random.normal(1000, 300, normal_samples),
        'Fwd Packet Length Mean': np.random.normal(500, 150, normal_samples),
        'Flow Bytes/s': np.random.normal(10000, 3000, normal_samples),
        'Flow Packets/s': np.random.normal(50, 15, normal_samples),
        'Fwd IAT Mean': np.random.normal(100000, 30000, normal_samples),
        'Bwd IAT Mean': np.random.normal(100000, 30000, normal_samples),
        'Label': ['BENIGN'] * normal_samples
    }
    
    # Attack Traffic (30%)
    attack_samples = num_samples - normal_samples
    attack_data = {
        'Flow Duration': np.random.normal(500000, 200000, attack_samples),
        'Total Fwd Packets': np.random.randint(50, 500, attack_samples),
        'Total Backward Packets': np.random.randint(0, 10, attack_samples),
        'Total Length of Fwd Packets': np.random.normal(5000, 1000, attack_samples),
        'Total Length of Bwd Packets': np.random.normal(100, 50, attack_samples),
        'Fwd Packet Length Mean': np.random.normal(1500, 500, attack_samples),
        'Flow Bytes/s': np.random.normal(50000, 10000, attack_samples),
        'Flow Packets/s': np.random.normal(200, 50, attack_samples),
        'Fwd IAT Mean': np.random.normal(10000, 5000, attack_samples),
        'Bwd IAT Mean': np.random.normal(10000, 5000, attack_samples),
        'Label': np.random.choice(['DDoS', 'DoS', 'PortScan', 'BruteForce'], attack_samples)
    }
    
    df_normal = pd.DataFrame(normal_data)
    df_attack = pd.DataFrame(attack_data)
    df = pd.concat([df_normal, df_attack], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def load_csv_data(file_path):
    """Load data from CSV file or use simulation"""
    
    if file_path and os.path.exists(file_path):
        try:
            st.info(f"📂 Loading data from: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            st.success(f"✅ Successfully loaded {len(df)} records from CSV!")
            return df, True
        except Exception as e:
            st.error(f"❌ Error loading CSV: {str(e)}")
            st.info("Using simulated data instead...")
            return generate_simulation_data(2000), False
    else:
        st.warning("⚠️ No CSV file found. Using simulated data instead.")
        st.info("📥 To use real data, download CIC-IDS2017 dataset and place CSV in Datasets folder")
        return generate_simulation_data(2000), False

def preprocess_data(df, is_real_csv=True):
    """Preprocess the dataset for training"""
    
    data = df.copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0)
    
    if 'Label' in data.columns:
        label_col = 'Label'
    elif ' Label' in data.columns:
        label_col = ' Label'
    else:
        st.error("❌ Label column not found!")
        return None, None, None
    
    data['Attack'] = data[label_col].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
    
    feature_columns = []
    possible_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
        'Flow Bytes/s', 'Flow Packets/s', 'Fwd IAT Mean',
        'Bwd IAT Mean', 'Fwd PSH Flags', 'Bwd PSH Flags',
        'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
        'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
        'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance'
    ]
    
    for feature in possible_features:
        if feature in data.columns:
            feature_columns.append(feature)
        elif f' {feature}' in data.columns:
            feature_columns.append(f' {feature}')
    
    if len(feature_columns) < 5:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_cols if col not in [label_col, 'Attack']]
    
    feature_columns = feature_columns[:15]
    
    if not feature_columns:
        st.error("❌ No valid features found!")
        return None, None, None
    
    X = data[feature_columns]
    y = data['Attack']

    return X, y, feature_columns

# --- Live Detection Simulation ---

SIM_ATTACK_OPTIONS = ['DDoS', 'DoS', 'PortScan', 'BruteForce']
SIM_FEED_CAP = 200
SIM_TIMELINE_CAP = 120
SIM_ALERT_CAP = 20
SIM_MAX_TICKS_PER_RUN = 300

def reset_simulation_state():
    """Reset all live simulation counters, feed and charts"""
    st.session_state['sim_running'] = False
    st.session_state['sim_tick'] = 0
    st.session_state['sim_total'] = 0
    st.session_state['sim_pred_normal'] = 0
    st.session_state['sim_pred_attack'] = 0
    st.session_state['sim_correct'] = 0
    st.session_state['sim_feed'] = []
    st.session_state['sim_timeline'] = []
    st.session_state['sim_alerts'] = []
    st.session_state['sim_attack_type_counts'] = {}

def generate_live_batch(batch_size, attack_intensity, attack_types, rng):
    """Generate one batch of simulated live flows with known true labels.
    Uses the same feature structure and distributions as generate_simulation_data()
    so the trained models see identical features."""
    if attack_types:
        n_attack = int(rng.binomial(batch_size, attack_intensity / 100.0))
    else:
        n_attack = 0
    n_normal = batch_size - n_attack

    frames = []
    if n_normal > 0:
        frames.append(pd.DataFrame({
            'Flow Duration': rng.normal(2000000, 500000, n_normal),
            'Total Fwd Packets': rng.integers(1, 50, n_normal),
            'Total Backward Packets': rng.integers(1, 50, n_normal),
            'Total Length of Fwd Packets': rng.normal(1000, 300, n_normal),
            'Total Length of Bwd Packets': rng.normal(1000, 300, n_normal),
            'Fwd Packet Length Mean': rng.normal(500, 150, n_normal),
            'Flow Bytes/s': rng.normal(10000, 3000, n_normal),
            'Flow Packets/s': rng.normal(50, 15, n_normal),
            'Fwd IAT Mean': rng.normal(100000, 30000, n_normal),
            'Bwd IAT Mean': rng.normal(100000, 30000, n_normal),
            'Label': ['BENIGN'] * n_normal
        }))
    if n_attack > 0:
        frames.append(pd.DataFrame({
            'Flow Duration': rng.normal(500000, 200000, n_attack),
            'Total Fwd Packets': rng.integers(50, 500, n_attack),
            'Total Backward Packets': rng.integers(0, 10, n_attack),
            'Total Length of Fwd Packets': rng.normal(5000, 1000, n_attack),
            'Total Length of Bwd Packets': rng.normal(100, 50, n_attack),
            'Fwd Packet Length Mean': rng.normal(1500, 500, n_attack),
            'Flow Bytes/s': rng.normal(50000, 10000, n_attack),
            'Flow Packets/s': rng.normal(200, 50, n_attack),
            'Fwd IAT Mean': rng.normal(10000, 5000, n_attack),
            'Bwd IAT Mean': rng.normal(10000, 5000, n_attack),
            'Label': rng.choice(attack_types, n_attack)
        }))

    batch = pd.concat(frames, ignore_index=True)
    batch = batch.iloc[rng.permutation(len(batch))].reset_index(drop=True)

    # Display-only fields for the live feed (not model features)
    batch['Source IP'] = [f"192.168.{rng.integers(0, 256)}.{rng.integers(1, 255)}" for _ in range(len(batch))]
    batch['Destination IP'] = [f"10.0.{rng.integers(0, 256)}.{rng.integers(1, 255)}" for _ in range(len(batch))]
    batch['Protocol'] = rng.choice(['TCP', 'UDP', 'ICMP'], len(batch), p=[0.7, 0.25, 0.05])

    return batch

def align_live_features(batch, feature_cols, X_reference):
    """Map generated flows onto the exact features the models were trained on,
    with the same cleaning as preprocess_data(). Features missing from the
    simulated flows (possible when trained on a real CSV) fall back to the
    training median, like the manual Live Detection tab."""
    X_live = pd.DataFrame(index=batch.index)
    for col in feature_cols:
        base = col.strip()
        if base in batch.columns:
            X_live[col] = batch[base].values
        else:
            X_live[col] = float(X_reference[col].median())
    X_live = X_live.replace([np.inf, -np.inf], np.nan)
    X_live = X_live.fillna(0)
    return X_live

def run_simulation_tick(model, feature_cols, X_reference, batch_size, attack_intensity, attack_types):
    """Generate one batch, classify it with the selected model and accumulate results"""
    rng = np.random.default_rng()
    batch = generate_live_batch(batch_size, attack_intensity, attack_types, rng)
    X_live = align_live_features(batch, feature_cols, X_reference)

    predictions = model.predict(X_live)
    probabilities = model.predict_proba(X_live)

    now = datetime.now().strftime("%H:%M:%S")
    attacks_this_tick = 0

    for i in range(len(batch)):
        true_label = str(batch['Label'].iloc[i])
        truth = 0 if 'BENIGN' in true_label.upper() else 1
        pred = int(predictions[i])
        confidence = float(probabilities[i][pred])

        st.session_state['sim_total'] += 1
        if pred == truth:
            st.session_state['sim_correct'] += 1

        if pred == 1:
            st.session_state['sim_pred_attack'] += 1
            attacks_this_tick += 1
            type_key = true_label if truth == 1 else 'False Alarm (BENIGN)'
            counts = st.session_state['sim_attack_type_counts']
            counts[type_key] = counts.get(type_key, 0) + 1

            if confidence >= 0.90:
                st.session_state['sim_alerts'].append({
                    'time': now,
                    'type': type_key,
                    'source': batch['Source IP'].iloc[i],
                    'destination': batch['Destination IP'].iloc[i],
                    'confidence': confidence
                })
        else:
            st.session_state['sim_pred_normal'] += 1

        st.session_state['sim_feed'].append({
            'Time': now,
            'Source': batch['Source IP'].iloc[i],
            'Destination': batch['Destination IP'].iloc[i],
            'Protocol': batch['Protocol'].iloc[i],
            'True Label': true_label,
            'Prediction': 'ATTACK' if pred == 1 else 'NORMAL',
            'Confidence': f"{confidence*100:.1f}%"
        })

    st.session_state['sim_timeline'].append({
        'Tick': st.session_state['sim_tick'] + 1,
        'Time': now,
        'Attacks Detected': attacks_this_tick,
        'Flows Analyzed': len(batch)
    })

    # Cap history so the UI stays light
    del st.session_state['sim_feed'][:-SIM_FEED_CAP]
    del st.session_state['sim_timeline'][:-SIM_TIMELINE_CAP]
    del st.session_state['sim_alerts'][:-SIM_ALERT_CAP]

    st.session_state['sim_tick'] += 1

def render_simulation_dashboard(model_name):
    """Draw the live simulation metrics, feed, charts and alerts from session state"""
    tick = st.session_state['sim_tick']
    total = st.session_state['sim_total']

    status = "🟢 RUNNING" if st.session_state['sim_running'] else "⏸️ PAUSED"
    st.caption(f"{status} | Tick: {tick} | Model: {model_name}")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🌐 Total Flows", total)
    with col2:
        st.metric("✅ Normal", st.session_state['sim_pred_normal'])
    with col3:
        st.metric("🚨 Attacks Detected", st.session_state['sim_pred_attack'])
    with col4:
        attack_rate = (st.session_state['sim_pred_attack'] / total * 100) if total > 0 else 0
        st.metric("⚠️ Attack Rate", f"{attack_rate:.1f}%")
    with col5:
        if total > 0:
            live_acc = st.session_state['sim_correct'] / total * 100
            st.metric("🎯 Live Accuracy", f"{live_acc:.2f}%")
        else:
            st.metric("🎯 Live Accuracy", "N/A")

    if total == 0:
        st.info("▶️ Press **Start** to begin streaming simulated traffic through the model.")
        return

    col_feed, col_dist = st.columns([3, 2])

    with col_feed:
        st.write(f"**📡 Live Traffic Feed (last {SIM_FEED_CAP} flows):**")

        feed_df = pd.DataFrame(st.session_state['sim_feed'][::-1])

        def highlight_prediction(row):
            if row['Prediction'] == 'ATTACK':
                return ['background-color: rgba(255, 107, 107, 0.35)'] * len(row)
            return ['background-color: rgba(78, 205, 196, 0.15)'] * len(row)

        st.dataframe(
            feed_df.style.apply(highlight_prediction, axis=1),
            use_container_width=True,
            hide_index=True,
            height=380,
            key=f"sim_feed_{tick}"
        )

    with col_dist:
        st.write("**📊 Detected Attacks by Type:**")

        type_counts = st.session_state['sim_attack_type_counts']
        if type_counts:
            dist_df = pd.DataFrame({
                'Attack Type': list(type_counts.keys()),
                'Count': list(type_counts.values())
            }).sort_values('Count', ascending=False)

            fig = px.bar(dist_df, x='Attack Type', y='Count',
                        color='Attack Type',
                        color_discrete_sequence=['#FF6B6B', '#FFA07A', '#45B7D1', '#4ECDC4', '#9b9b9b'])
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key=f"sim_dist_{tick}")
        else:
            st.info("No attacks detected yet.")

    st.write("**📈 Attack Timeline:**")

    timeline_df = pd.DataFrame(st.session_state['sim_timeline'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline_df['Tick'], y=timeline_df['Attacks Detected'],
        mode='lines+markers', name='Attacks Detected',
        line=dict(color='#FF6B6B', width=2), fill='tozeroy'
    ))
    fig.add_trace(go.Scatter(
        x=timeline_df['Tick'], y=timeline_df['Flows Analyzed'],
        mode='lines', name='Flows Analyzed',
        line=dict(color='#4ECDC4', width=1, dash='dot')
    ))
    fig.update_layout(
        title=f'Attacks Over Time (last {SIM_TIMELINE_CAP} ticks)',
        xaxis_title='Tick', yaxis_title='Flows', height=300
    )
    st.plotly_chart(fig, use_container_width=True, key=f"sim_timeline_{tick}")

    st.write("**🚨 Recent High-Confidence Alerts (≥90%):**")

    alerts = st.session_state['sim_alerts']
    if alerts:
        for alert in alerts[::-1][:5]:
            st.error(
                f"🚨 [{alert['time']}] **{alert['type']}** — "
                f"{alert['source']} → {alert['destination']} — "
                f"Confidence: {alert['confidence']*100:.1f}%"
            )
        st.caption("📧 Simulated response: alert email queued & incident report generated (no real emails are sent)")
    else:
        st.success("✅ No high-confidence attack alerts yet.")

def train_models(X, y, selected_models):
    """Train multiple ML models"""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    models = {}
    results = {}
    
    model_configs = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, model_name in enumerate(selected_models):
        status_text.text(f"🤖 Training {model_name}...")
        
        model = model_configs[model_name]
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        
        models[model_name] = model
        results[model_name] = {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        progress_bar.progress((idx + 1) / len(selected_models))
    
    status_text.text("✅ All models trained successfully!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return models, results, X_test, y_test, X_train

# Main Application
def main():
    
    st.sidebar.subheader("📁 Dataset Selection")
    
    if os.path.exists("Datasets"):
        csv_files = [f for f in os.listdir("Datasets") if f.endswith('.csv')]
    else:
        csv_files = []
    
    if csv_files:
        selected_file = st.sidebar.selectbox("Select CSV File:", csv_files)
        selected_file_path = os.path.join("Datasets", selected_file)
    else:
        st.sidebar.warning("No CSV files found")
        st.sidebar.info("Place CSV in Datasets folder")
        selected_file = None
        selected_file_path = None
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Selection")
    
    model_options = ['Random Forest', 'Gradient Boosting', 'SVM', 'Neural Network']
    selected_models = st.sidebar.multiselect(
        "Select Models to Train:",
        model_options,
        default=['Random Forest', 'Gradient Boosting']
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Training Options")
    
    data_size = st.sidebar.slider("Simulated Data Size:", 1000, 5000, 2000, 500)
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 Train Models Now", use_container_width=True):
        if not selected_models:
            st.error("⚠️ Please select at least one model!")
            return
        
        st.session_state['model_trained'] = True
        st.session_state['train_file_path'] = selected_file_path
        st.session_state['selected_models'] = selected_models
        st.session_state['data_size'] = data_size
        st.rerun()
    
    if st.session_state.get('model_trained', False):
        
        train_file = st.session_state.get('train_file_path', None)
        data, is_real = load_csv_data(train_file)
        
        if not is_real:
            data = generate_simulation_data(st.session_state.get('data_size', 2000))
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Dataset Overview",
            "🤖 Model Performance",
            "📈 Advanced Analytics",
            "🔴 Live Detection",
            "⚡ Live Simulation",
            "📜 History & Reports"
        ])
        
        with tab1:
            st.subheader("📁 Dataset Information")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("📦 Total Records", len(data))
            with col2:
                st.metric("📊 Features", len(data.columns) - 1)
            with col3:
                if 'Label' in data.columns:
                    benign_count = len(data[data['Label'].str.contains('BENIGN', case=False, na=False)])
                elif ' Label' in data.columns:
                    benign_count = len(data[data[' Label'].str.contains('BENIGN', case=False, na=False)])
                else:
                    benign_count = 0
                st.metric("✅ Normal Traffic", benign_count)
            with col4:
                attack_count = len(data) - benign_count
                st.metric("🚨 Attack Traffic", attack_count)
            with col5:
                attack_ratio = (attack_count / len(data)) * 100 if len(data) > 0 else 0
                st.metric("⚠️ Attack Ratio", f"{attack_ratio:.1f}%")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Sample Data Preview:**")
                st.dataframe(data.head(10), use_container_width=True)
            
            with col2:
                if 'Label' in data.columns or ' Label' in data.columns:
                    label_col = 'Label' if 'Label' in data.columns else ' Label'
                    st.write("**Attack Types Distribution:**")
                    
                    attack_dist = data[label_col].value_counts()
                    
                    fig = px.pie(
                        values=attack_dist.values, 
                        names=attack_dist.index,
                        title="Traffic Distribution",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Preprocess data
        result = preprocess_data(data, is_real)
        
        if result[0] is None:
            st.error("Failed to preprocess data!")
            return
        
        X, y, feature_cols = result
        
        selected_models = st.session_state.get('selected_models', ['Random Forest'])
        models, results, X_test, y_test, X_train = train_models(X, y, selected_models)
        
        st.session_state['models'] = models
        st.session_state['results'] = results
        st.session_state['feature_cols'] = feature_cols
        st.session_state['X_data'] = X
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for model_name, result in results.items():
            st.session_state['training_history'].append({
                'timestamp': timestamp,
                'model': model_name,
                'accuracy': result['accuracy'],
                'samples': len(X)
            })
        
        with tab2:
            st.subheader("🤖 Model Performance Comparison")
            
            col1, col2, col3, col4 = st.columns(4)
            
            best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
            
            with col1:
                st.metric("🏆 Best Model", best_model[0])
            with col2:
                st.metric("🎯 Best Accuracy", f"{best_model[1]['accuracy']*100:.2f}%")
            with col3:
                avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
                st.metric("📊 Avg Accuracy", f"{avg_accuracy*100:.2f}%")
            with col4:
                st.metric("🔢 Models Trained", len(results))
            
            # Model comparison bar chart
            st.write("**Accuracy Comparison:**")
            model_names = list(results.keys())
            accuracies = [results[m]['accuracy'] * 100 for m in model_names]
            
            fig = go.Figure(data=[
                go.Bar(x=model_names, y=accuracies, 
                       text=[f"{acc:.2f}%" for acc in accuracies],
                       textposition='auto',
                       marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
            ])
            fig.update_layout(
                title="Model Accuracy Comparison",
                xaxis_title="Model",
                yaxis_title="Accuracy (%)",
                yaxis_range=[0, 100],
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics for each model
            st.write("**Detailed Performance Metrics:**")
            
            for model_name, result in results.items():
                with st.expander(f"📊 {model_name} - Detailed Metrics", expanded=(model_name==best_model[0])):
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    cm = result['confusion_matrix']
                    tn, fp, fn, tp = cm.ravel()
                    
                    accuracy = result['accuracy']
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    with col1:
                        st.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")
                    with col2:
                        st.metric("🔍 Precision", f"{precision*100:.2f}%")
                    with col3:
                        st.metric("📊 Recall", f"{recall*100:.2f}%")
                    with col4:
                        st.metric("⚡ F1-Score", f"{f1*100:.2f}%")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Confusion Matrix
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                                    xticklabels=['Normal', 'Attack'],
                                    yticklabels=['Normal', 'Attack'],
                                    cbar_kws={'label': 'Count'})
                        plt.ylabel('Actual')
                        plt.xlabel('Predicted')
                        plt.title(f'{model_name} - Confusion Matrix')
                        st.pyplot(fig)
                    
                    with col2:
                        # Metrics bar chart
                        metrics_data = pd.DataFrame({
                            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                            'Score': [accuracy*100, precision*100, recall*100, f1*100]
                        })
                        
                        fig = px.bar(metrics_data, x='Metric', y='Score',
                                    title=f'{model_name} - Performance Metrics',
                                    color='Metric',
                                    text='Score',
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                        fig.update_layout(yaxis_range=[0, 105], showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("📈 Advanced Analytics & Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**ROC Curves Comparison:**")
                
                fig = go.Figure()
                
                for model_name, result in results.items():
                    fig.add_trace(go.Scatter(
                        x=result['fpr'], 
                        y=result['tpr'],
                        name=f"{model_name} (AUC={result['roc_auc']:.3f})",
                        mode='lines',
                        line=dict(width=2)
                    ))
                
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name='Random Classifier',
                    mode='lines',
                    line=dict(dash='dash', color='gray')
                ))
                
                fig.update_layout(
                    title='Receiver Operating Characteristic (ROC) Curves',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Precision-Recall Curves:**")
                
                fig = go.Figure()
                
                for model_name, result in results.items():
                    fig.add_trace(go.Scatter(
                        x=result['recall'], 
                        y=result['precision'],
                        name=model_name,
                        mode='lines',
                        line=dict(width=2)
                    ))
                
                fig.update_layout(
                    title='Precision-Recall Curves',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Feature Importance Analysis:**")
            
            # Get feature importance from Random Forest
            if 'Random Forest' in models:
                rf_model = models['Random Forest']
                importance = rf_model.feature_importances_
                
                feat_imp_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = px.bar(feat_imp_df, x='Importance', y='Feature',
                                orientation='h',
                                title='Feature Importance (Random Forest)',
                                color='Importance',
                                color_continuous_scale='Viridis')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Top 5 Features:**")
                    for idx, row in feat_imp_df.head(5).iterrows():
                        st.metric(
                            row['Feature'][:20],
                            f"{row['Importance']:.4f}",
                            delta=None
                        )
            
            st.write("**Feature Correlation Heatmap:**")
            
            corr_matrix = X[feature_cols[:10]].corr()
            
            fig = px.imshow(corr_matrix,
                           labels=dict(color="Correlation"),
                           x=corr_matrix.columns,
                           y=corr_matrix.columns,
                           color_continuous_scale='RdBu_r',
                           aspect="auto")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("🔴 Live Traffic Detection & Analysis")
            
            # Real-time counters
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("✅ Normal Detected", st.session_state['prediction_count']['normal'])
            with col2:
                st.metric("🚨 Attacks Detected", st.session_state['prediction_count']['attack'])
            with col3:
                total_pred = sum(st.session_state['prediction_count'].values())
                st.metric("📊 Total Predictions", total_pred)
            with col4:
                if total_pred > 0:
                    attack_pct = (st.session_state['prediction_count']['attack'] / total_pred) * 100
                    st.metric("⚠️ Attack Rate", f"{attack_pct:.1f}%")
                else:
                    st.metric("⚠️ Attack Rate", "0%")
            
            st.markdown("---")
            
            # Model selection for prediction
            selected_pred_model = st.selectbox(
                "Select Model for Prediction:",
                list(models.keys()),
                key="pred_model"
            )
            
            st.write("**Adjust Network Parameters:**")
            
            input_data = {}
            X_data = st.session_state['X_data']
            
            cols = st.columns(3)
            for idx, feature in enumerate(feature_cols[:9]):
                with cols[idx % 3]:
                    sample_val = float(X_data[feature].median())
                    min_val = float(X_data[feature].min())
                    max_val = float(X_data[feature].max())
                    
                    input_data[feature] = st.slider(
                        feature.strip()[:30],
                        min_value=min_val,
                        max_value=max_val,
                        value=sample_val,
                        key=f"slider_{feature}"
                    )
            
            for feature in feature_cols[9:]:
                input_data[feature] = float(X_data[feature].median())
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🔍 Analyze Traffic", use_container_width=True):
                    test_df = pd.DataFrame([input_data])
                    
                    model = models[selected_pred_model]
                    prediction = model.predict(test_df)[0]
                    probability = model.predict_proba(test_df)[0]
                    
                    # Update counters
                    if prediction == 0:
                        st.session_state['prediction_count']['normal'] += 1
                    else:
                        st.session_state['prediction_count']['attack'] += 1
                    
                    st.markdown("### 🎯 Detection Result")
                    
                    if prediction == 0:
                        st.success(f"✅ **NORMAL TRAFFIC** - No threat detected")
                        st.info(f"🔒 Confidence: {probability[0]*100:.2f}%")
                        st.write("**Status:** Traffic appears legitimate. Continue monitoring.")
                    else:
                        st.error(f"🚨 **ATTACK DETECTED** - Potential intrusion!")
                        st.warning(f"⚠️ Threat Probability: {probability[1]*100:.2f}%")
                        st.write("**Recommended Actions:**")
                        st.write("- 🛑 Block source IP immediately")
                        st.write("- 📧 Alert security team")
                        st.write("- 📝 Log incident for analysis")
                        st.write("- 🔍 Investigate traffic patterns")
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability[1]*100,
                        title={'text': "Attack Probability"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Multi-Model Consensus:**")
                
                if st.button("🔬 Analyze with All Models", use_container_width=True):
                    test_df = pd.DataFrame([input_data])
                    
                    consensus_results = []
                    
                    for model_name, model in models.items():
                        pred = model.predict(test_df)[0]
                        prob = model.predict_proba(test_df)[0]
                        
                        consensus_results.append({
                            'Model': model_name,
                            'Prediction': 'Attack' if pred == 1 else 'Normal',
                            'Confidence': f"{max(prob)*100:.1f}%",
                            'Attack_Prob': prob[1]
                        })
                    
                    consensus_df = pd.DataFrame(consensus_results)
                    
                    # Visual representation
                    fig = px.bar(consensus_df, x='Model', y='Attack_Prob',
                                color='Prediction',
                                title='Model Predictions Comparison',
                                labels={'Attack_Prob': 'Attack Probability'},
                                color_discrete_map={'Attack': '#FF6B6B', 'Normal': '#4ECDC4'})
                    fig.update_layout(yaxis_range=[0, 1], height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(consensus_df[['Model', 'Prediction', 'Confidence']], 
                               use_container_width=True, hide_index=True)
                    
                    # Majority voting
                    attack_count = sum(1 for r in consensus_results if r['Prediction'] == 'Attack')
                    majority = "Attack" if attack_count > len(models)/2 else "Normal"
                    
                    st.info(f"🗳️ **Majority Vote:** {majority} ({attack_count}/{len(models)} models)")

        with tab5:
            st.subheader("⚡ Live Detection Simulation")

            st.markdown("""
            Continuously streams **simulated network traffic** (normal + attack flows with known
            true labels) through a trained model and updates detections, charts and alerts live.
            """)

            ctrl1, ctrl2, ctrl3 = st.columns(3)

            with ctrl1:
                sim_model_name = st.selectbox(
                    "Detection Model:",
                    list(models.keys()),
                    key="sim_model"
                )
            with ctrl2:
                sim_interval = st.slider("Refresh Interval (seconds):", 0.5, 5.0, 1.0, 0.5, key="sim_interval")
            with ctrl3:
                sim_batch_size = st.slider("Flows per Tick:", 1, 20, 5, key="sim_batch_size")

            ctrl4, ctrl5 = st.columns([1, 2])

            with ctrl4:
                sim_intensity = st.slider("Attack Intensity (%):", 0, 100, 30, 5, key="sim_intensity")
            with ctrl5:
                sim_attack_types = st.multiselect(
                    "Attack Types:",
                    SIM_ATTACK_OPTIONS,
                    default=SIM_ATTACK_OPTIONS,
                    key="sim_attack_types"
                )

            if sim_intensity > 0 and not sim_attack_types:
                st.warning("⚠️ No attack types selected — the stream will contain only normal traffic.")

            btn1, btn2, btn3 = st.columns(3)

            with btn1:
                if st.button("▶️ Start", use_container_width=True, key="sim_start"):
                    st.session_state['sim_running'] = True
            with btn2:
                if st.button("⏸️ Pause", use_container_width=True, key="sim_pause"):
                    st.session_state['sim_running'] = False
            with btn3:
                if st.button("🔄 Reset", use_container_width=True, key="sim_reset"):
                    reset_simulation_state()

            st.markdown("---")

            sim_view = st.empty()

            if st.session_state['sim_running']:
                # In-run loop: placeholders update without reruns, so models are NOT
                # retrained per tick. Any button/slider interaction interrupts the loop
                # and triggers a normal rerun. Bounded so the run can't block forever.
                sim_model = models[sim_model_name]
                for _ in range(SIM_MAX_TICKS_PER_RUN):
                    if not st.session_state['sim_running']:
                        break
                    run_simulation_tick(
                        sim_model, feature_cols, X,
                        sim_batch_size, sim_intensity, sim_attack_types
                    )
                    with sim_view.container():
                        render_simulation_dashboard(sim_model_name)
                    time.sleep(sim_interval)
                else:
                    st.session_state['sim_running'] = False
                    st.info(f"⏸️ Auto-paused after {SIM_MAX_TICKS_PER_RUN} ticks — press ▶️ Start to continue.")
            else:
                with sim_view.container():
                    render_simulation_dashboard(sim_model_name)

        with tab6:
            st.subheader("📜 Training History & Report Generation")
            
            if st.session_state['training_history']:
                st.write("**Recent Training Sessions:**")
                
                history_df = pd.DataFrame(st.session_state['training_history'])
                history_df['accuracy_pct'] = history_df['accuracy'] * 100
                
                st.dataframe(
                    history_df[['timestamp', 'model', 'accuracy_pct', 'samples']].tail(10),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'timestamp': 'Training Time',
                        'model': 'Model',
                        'accuracy_pct': st.column_config.NumberColumn('Accuracy (%)', format="%.2f"),
                        'samples': 'Samples'
                    }
                )
                
                st.write("**Accuracy Trend Over Time:**")
                
                fig = px.line(history_df.tail(20), x='timestamp', y='accuracy_pct', 
                             color='model', markers=True,
                             title='Model Performance Over Time',
                             labels={'accuracy_pct': 'Accuracy (%)', 'timestamp': 'Time'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No training history available yet.")
            
            st.markdown("---")
            st.write("**📊 Generate Comprehensive Report:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    st.info("PDF report generation feature - Coming soon!")
            
            with col2:
                if st.button("📊 Export Results CSV", use_container_width=True):
                    # Create results dataframe
                    results_data = []
                    for model_name, result in results.items():
                        cm = result['confusion_matrix']
                        tn, fp, fn, tp = cm.ravel()
                        
                        results_data.append({
                            'Model': model_name,
                            'Accuracy': result['accuracy'],
                            'ROC_AUC': result['roc_auc'],
                            'True_Positives': tp,
                            'True_Negatives': tn,
                            'False_Positives': fp,
                            'False_Negatives': fn
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    csv = results_df.to_csv(index=False)
                    
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"nids_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if st.button("🔄 Clear History", use_container_width=True):
                    st.session_state['training_history'] = []
                    st.session_state['prediction_count'] = {'normal': 0, 'attack': 0}
                    st.success("✅ History cleared!")
                    st.rerun()
            
            st.markdown("---")
            st.write("**📊 System Statistics:**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎓 Total Trainings", len(st.session_state['training_history']))
            with col2:
                if st.session_state['training_history']:
                    avg_acc = np.mean([h['accuracy'] for h in st.session_state['training_history']])
                    st.metric("📈 Avg Accuracy", f"{avg_acc*100:.2f}%")
                else:
                    st.metric("📈 Avg Accuracy", "N/A")
            with col3:
                total_predictions = sum(st.session_state['prediction_count'].values())
                st.metric("🔍 Total Predictions", total_predictions)
            with col4:
                st.metric("🗂️ Dataset Size", len(data))
    
    else:
        st.info("👈 **Configure settings in sidebar and click 'Train Models Now' to begin**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🤖 Multi-Model Support
            - Random Forest
            - Gradient Boosting
            - Support Vector Machine
            - Neural Network
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Advanced Analytics
            - ROC & PR Curves
            - Feature Importance
            - Correlation Analysis
            - Real-time Metrics
            """)
        
        with col3:
            st.markdown("""
            ### 🎯 Professional Features
            - Multi-model Consensus
            - Training History
            - Export Reports
            - Live Detection
            """)
        
        st.markdown("---")
        st.info("⚡ **Live Detection Simulation** needs a trained model first. In a hurry? Quick-train one on simulated data:")

        if st.button("⚡ Quick Train & Enable Live Simulation"):
            st.session_state['model_trained'] = True
            st.session_state['train_file_path'] = None
            st.session_state['selected_models'] = ['Random Forest']
            st.session_state['data_size'] = 2000
            st.rerun()

        st.markdown("---")
        st.subheader("📖 Quick Start Guide")
        
        st.markdown("""
        1. **📁 Dataset Setup:**
           - Download CIC-IDS2017 dataset from [UNB Website](https://www.unb.ca/cic/datasets/ids-2017.html)
           - Create `Datasets` folder in your project directory
           - Place CSV files in the `Datasets` folder
           - Or use simulated data for testing
        
        2. **🤖 Model Configuration:**
           - Select one or more ML models from sidebar
           - Adjust simulated data size if needed
           - Click "🚀 Train Models Now"
        
        3. **📊 Analysis:**
           - View dataset overview and statistics
           - Compare model performances
           - Explore advanced analytics and visualizations
        
        4. **🔴 Live Detection:**
           - Test with custom network parameters
           - Get multi-model predictions
           - Track detection statistics
        
        5. **📜 Reports:**
           - View training history
           - Export results to CSV
           - Generate comprehensive reports
        """)
        
        st.markdown("---")
        st.info("💡 **Tip:** Start with Random Forest and Gradient Boosting for best performance!")

if __name__ == "__main__":
    main()