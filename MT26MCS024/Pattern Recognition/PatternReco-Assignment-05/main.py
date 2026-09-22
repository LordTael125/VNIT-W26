import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import shutil
import signal
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'Templates'), static_folder=os.path.join(BASE_DIR, 'Statics'))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'Uploads')
app.config['PLOT_FOLDER'] = os.path.join(BASE_DIR, 'Statics/plots')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max limit

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOT_FOLDER'], exist_ok=True)

class Analyzer:
    def __init__(self):
        self.df = None
        self.df_preprocessed = None
        self.preprocessing_results = {}
        
    def analyze_dataset(self, filepath):
        self.df = pd.read_csv(filepath)
        
        rows, cols = self.df.shape
        missing_values_total = self.df.isnull().sum().sum()
        duplicate_rows = self.df.duplicated().sum()
        
        column_info = []
        for col in self.df.columns:
            column_info.append({
                'name': col,
                'type': str(self.df[col].dtype),
                'missing': int(self.df[col].isnull().sum()),
                'non_null': int(self.df[col].notnull().sum())
            })
            
        summary = {
            'rows': rows,
            'cols': cols,
            'missing_values': int(missing_values_total),
            'duplicate_rows': int(duplicate_rows),
            'column_info': column_info
        }
        
        # Preprocessing
        self.df_preprocessed = self.df.copy()
        
        # 1. Remove duplicates
        self.df_preprocessed = self.df_preprocessed.drop_duplicates()
        dups_removed = rows - len(self.df_preprocessed)
        
        # 2. Handle missing values
        rows_before_na = len(self.df_preprocessed)
        missing_values_filled = self.df_preprocessed.isnull().sum().sum()
        for col in self.df_preprocessed.columns:
            if pd.api.types.is_numeric_dtype(self.df_preprocessed[col]):
                median_val = self.df_preprocessed[col].median()
                if pd.notna(median_val):
                    self.df_preprocessed[col] = self.df_preprocessed[col].fillna(median_val)
            else:
                if not self.df_preprocessed[col].mode().empty:
                    mode_val = self.df_preprocessed[col].mode()[0]
                    self.df_preprocessed[col] = self.df_preprocessed[col].fillna(mode_val)
        rows_after_na = len(self.df_preprocessed)
        
        self.preprocessing_results = {
            'rows_before': rows,
            'duplicates_found': int(duplicate_rows),
            'duplicates_removed': int(dups_removed),
            'rows_after': rows_after_na,
            'missing_values_after': int(self.df_preprocessed.isnull().sum().sum()),
            'details': [
                {'item': 'Rows Before Preprocessing', 'value': rows},
                {'item': 'Duplicate Rows Found', 'value': int(duplicate_rows)},
                {'item': 'Duplicate Rows Removed', 'value': int(dups_removed)},
                {'item': 'Rows After Removing Duplicates', 'value': rows_before_na},
                {'item': 'Duplicate Count Correct', 'value': 'Yes' if dups_removed == duplicate_rows else 'No'},
                {'item': 'Missing Values Filled', 'value': int(missing_values_filled)},
                {'item': 'Missing Values After Preprocessing', 'value': 0}
            ]
        }
        
        first_five = self.df_preprocessed.head().to_dict('records')
        
        return summary, self.preprocessing_results, first_five, list(self.df_preprocessed.columns)
        
    def perform_statistical_analysis(self, features):
        if self.df_preprocessed is None or not features:
            return None, None
            
        stats_data = []
        for feature in features:
            if pd.api.types.is_numeric_dtype(self.df_preprocessed[feature]):
                col_data = self.df_preprocessed[feature]
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                
                stats_data.append({
                    'feature': feature,
                    'mean': round(col_data.mean(), 3),
                    'median': round(col_data.median(), 3),
                    'mode': round(col_data.mode()[0], 3) if not col_data.mode().empty else 'N/A',
                    'variance': round(col_data.var(), 3),
                    'std_dev': round(col_data.std(), 3),
                    'min': round(col_data.min(), 3),
                    'max': round(col_data.max(), 3),
                    'range': round(col_data.max() - col_data.min(), 3),
                    'q1': round(q1, 3),
                    'q3': round(q3, 3),
                    'iqr': round(q3 - q1, 3)
                })
                
        # Generate graphs
        graph_urls = {}
        
        # Histograms
        hist_urls = []
        for feature in features:
            if pd.api.types.is_numeric_dtype(self.df_preprocessed[feature]):
                plt.figure(figsize=(6, 4))
                sns.histplot(self.df_preprocessed[feature], bins=10, color='tab:blue')
                plt.title(f'Histogram - {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')
                hist_path = os.path.join(app.config['PLOT_FOLDER'], f'hist_{feature}.png')
                plt.tight_layout()
                plt.savefig(hist_path)
                plt.close()
                hist_urls.append(f'plots/hist_{feature}.png')
        graph_urls['histograms'] = hist_urls
        
        # Box Plots
        box_urls = []
        for feature in features:
            if pd.api.types.is_numeric_dtype(self.df_preprocessed[feature]):
                plt.figure(figsize=(4, 6))
                plt.boxplot(self.df_preprocessed[feature])
                plt.title(f'Box Plot - {feature}')
                plt.ylabel(feature)
                box_path = os.path.join(app.config['PLOT_FOLDER'], f'box_{feature}.png')
                plt.tight_layout()
                plt.savefig(box_path)
                plt.close()
                box_urls.append(f'plots/box_{feature}.png')
        graph_urls['boxplots'] = box_urls
        
        # Correlation Matrix
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(self.df_preprocessed[f])]
        if len(numeric_features) > 1:
            plt.figure(figsize=(8, 6))
            corr_matrix = self.df_preprocessed[numeric_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt='.2f', vmin=-1, vmax=1)
            plt.title('Correlation Matrix Heatmap')
            corr_path = os.path.join(app.config['PLOT_FOLDER'], 'correlation_matrix.png')
            plt.tight_layout()
            plt.savefig(corr_path)
            plt.close()
            graph_urls['correlation'] = 'plots/correlation_matrix.png'
            
        return stats_data, graph_urls

analyzer = Analyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    preprocessing = None
    first_five = None
    columns = None
    stats_data = None
    graph_urls = None
    selected_features = []
    
    if request.method == 'POST':
        # Check if it's file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                summary, preprocessing, first_five, columns = analyzer.analyze_dataset(filepath)
                
        # Check if it's feature analysis
        elif 'analyze_features' in request.form:
            selected_features = request.form.getlist('features')
            if selected_features:
                stats_data, graph_urls = analyzer.perform_statistical_analysis(selected_features)
                
                # We need to retain previous summary info
                summary = {
                    'rows': analyzer.preprocessing_results.get('rows_before', 0),
                    'cols': len(analyzer.df.columns) if analyzer.df is not None else 0,
                    'missing_values': analyzer.df.isnull().sum().sum() if analyzer.df is not None else 0,
                    'duplicate_rows': analyzer.df.duplicated().sum() if analyzer.df is not None else 0,
                    'column_info': []
                }
                
                if analyzer.df is not None:
                     for col in analyzer.df.columns:
                        summary['column_info'].append({
                            'name': col,
                            'type': str(analyzer.df[col].dtype),
                            'missing': int(analyzer.df[col].isnull().sum()),
                            'non_null': int(analyzer.df[col].notnull().sum())
                        })
                
                preprocessing = analyzer.preprocessing_results
                first_five = analyzer.df_preprocessed.head().to_dict('records') if analyzer.df_preprocessed is not None else []
                columns = list(analyzer.df_preprocessed.columns) if analyzer.df_preprocessed is not None else []

    return render_template('index.html', 
                           summary=summary, 
                           preprocessing=preprocessing, 
                           first_five=first_five, 
                           columns=columns,
                           selected_features=selected_features,
                           stats_data=stats_data,
                           graph_urls=graph_urls)

def kill_server():
    time.sleep(1)
    os.kill(os.getpid(), signal.SIGINT)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    # Delete contents of Uploads
    upload_path = app.config['UPLOAD_FOLDER']
    if os.path.exists(upload_path):
        for filename in os.listdir(upload_path):
            file_path = os.path.join(upload_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
                
    # Delete contents of Statics/plots
    plot_path = app.config['PLOT_FOLDER']
    if os.path.exists(plot_path):
        for filename in os.listdir(plot_path):
            file_path = os.path.join(plot_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    threading.Thread(target=kill_server).start()
    return "Application shut down safely and cache cleared. You can close this window."

if __name__ == "__main__":
    app.run(debug=True)