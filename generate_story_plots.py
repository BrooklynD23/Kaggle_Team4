import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.data.feature_engineering import StudentFeatureEngineer

# Setup
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "dataset.csv"
PLOTS_DIR = PROJECT_ROOT / "artifacts" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_and_process_data():
    print("📂 Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    print("🔧 Applying feature engineering...")
    engineer = StudentFeatureEngineer()
    df_engineered = engineer.fit_transform(df)
    
    return df_engineered

def plot_ghosting_effect(df):
    print("📊 Generating Ghosting Effect plot...")
    plt.figure(figsize=(10, 6))
    
    # Option 1: Average (Mean) Bar Chart
    # Using barplot to show the mean with confidence intervals
    ax = sns.barplot(
        x='Target', 
        y='units_without_eval_sem1', 
        data=df, 
        palette='viridis',
        estimator=np.mean
    )
    
    plt.title('The "Ghosting" Effect: Average Units Without Evaluation vs. Student Outcome')
    plt.ylabel('Average Units Without Evaluation (Sem 1)')
    plt.xlabel('Student Outcome')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "story_ghosting_effect_average.png", dpi=300)
    plt.close()

def plot_ghosting_prevalence(df):
    print("📊 Generating Ghosting Prevalence plot...")
    plt.figure(figsize=(10, 6))
    
    # Create a binary flag for ghosting (having > 0 unevaluated units)
    df['has_ghosted'] = df['units_without_eval_sem1'] > 0
    
    # Calculate percentage of students who ghosted for each target group
    # We group by Target and calculate the mean of the boolean flag (which gives the proportion)
    # Then multiply by 100 to get percentage
    prevalence = df.groupby('Target')['has_ghosted'].mean().reset_index()
    prevalence['has_ghosted_pct'] = prevalence['has_ghosted'] * 100
    
    ax = sns.barplot(
        x='Target', 
        y='has_ghosted_pct', 
        data=prevalence, 
        palette='viridis'
    )
    
    plt.title('The "Ghosting" Risk: % of Students with >0 Unevaluated Units')
    plt.ylabel('% of Students')
    plt.xlabel('Student Outcome')
    plt.ylim(0, 100)  # Set y-axis to 0-100%
    
    # Add percentage labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "story_ghosting_effect_prevalence.png", dpi=300)
    plt.close()

def plot_financial_impact(df):
    print("📊 Generating Financial Impact plot...")
    
    # Calculate percentages
    cross_tab = pd.crosstab(df['Tuition fees up to date'], df['Target'], normalize='index') * 100
    
    ax = cross_tab.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    
    plt.title('Impact of Tuition Payment Status on Student Success')
    plt.xlabel('Tuition Fees Up to Date (0=No, 1=Yes)')
    plt.ylabel('Percentage of Students (%)')
    plt.legend(title='Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks([0, 1], ['Arrears', 'Up to Date'], rotation=0)
    
    # Add percentage labels
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "story_financial_impact.png", dpi=300)
    plt.close()

def plot_academic_momentum(df):
    print("📊 Generating Academic Momentum plot...")
    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(
        data=df, 
        x='Curricular units 1st sem (grade)', 
        y='Curricular units 2nd sem (grade)', 
        hue='Target', 
        style='Target',
        palette='viridis',
        alpha=0.6
    )
    
    # Add diagonal line (No change line)
    max_val = max(df['Curricular units 1st sem (grade)'].max(), df['Curricular units 2nd sem (grade)'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='No Grade Change')
    
    plt.title('Academic Momentum: 1st vs 2nd Semester Grades')
    plt.xlabel('1st Semester Grade')
    plt.ylabel('2nd Semester Grade')
    plt.legend(title='Outcome')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "story_academic_momentum.png", dpi=300)
    plt.close()

def plot_correlation_heatmap(df):
    print("📊 Generating Correlation Heatmap...")
    
    # Encode Target
    target_map = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    df_corr = df.copy()
    df_corr['Target_Encoded'] = df_corr['Target'].map(target_map)
    
    # Select features to correlate (numeric only)
    numeric_df = df_corr.select_dtypes(include=[np.number])
    
    # Calculate correlation with Target
    correlations = numeric_df.corr()['Target_Encoded'].sort_values(ascending=False)
    
    # Top 10 positive and Top 5 negative
    top_pos = correlations.head(11).index.tolist() # Includes Target itself
    top_neg = correlations.tail(5).index.tolist()
    
    cols_to_plot = top_pos + top_neg
    # Remove duplicates if any
    cols_to_plot = list(dict.fromkeys(cols_to_plot))
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(numeric_df[cols_to_plot].corr(), dtype=bool))
    
    sns.heatmap(
        numeric_df[cols_to_plot].corr(), 
        mask=mask, 
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm', 
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )
    
    plt.title('Correlation of Top Features with Student Outcome')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "story_correlation_heatmap.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    try:
        df = load_and_process_data()
        plot_ghosting_effect(df)
        plot_ghosting_prevalence(df)
        plot_financial_impact(df)
        plot_academic_momentum(df)
        plot_correlation_heatmap(df)
        print("✅ All story plots generated successfully in artifacts/plots/")
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
