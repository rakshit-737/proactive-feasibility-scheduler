import os
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title='Proactive Scheduler Dashboard', layout='wide')
st.title('Proactive Feasibility Scheduler — Interactive Dashboard')
st.caption('Model comparisons, SHAP outputs, fairness, scaling, traces, and ROI summaries.')

paths = {
    'Ablation': os.path.join(PROJECT_ROOT, '05_results', 'models', 'ablation_study_results.csv'),
    'Scheduler Benchmark': os.path.join(PROJECT_ROOT, '05_results', 'schedulers', 'multi_scheduler_benchmark.csv'),
    'Fairness': os.path.join(PROJECT_ROOT, '05_results', 'fairness', 'fairness_metrics.csv'),
    'Scaling': os.path.join(PROJECT_ROOT, '05_results', 'scaling', 'scaling_analysis.csv'),
    'Trace Validation': os.path.join(PROJECT_ROOT, '05_results', 'traces', 'lanl_validation_results.csv'),
    'ROI': os.path.join(PROJECT_ROOT, '05_results', 'roi', 'cost_benefit_analysis.csv'),
}

for label, path in paths.items():
    st.subheader(label)
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.dataframe(df, use_container_width=True)
    else:
        st.info(f'Run pipeline to generate: {path}')

st.subheader('SHAP Visualizations')
shap_dir = os.path.join(PROJECT_ROOT, '05_results', 'shap')
if os.path.isdir(shap_dir):
    imgs = [f for f in sorted(os.listdir(shap_dir)) if f.endswith('.png')]
    if imgs:
        choice = st.selectbox('Select SHAP image', imgs)
        st.image(os.path.join(shap_dir, choice), use_container_width=True)
    else:
        st.info('No SHAP images found yet.')
else:
    st.info('SHAP directory not found yet.')
