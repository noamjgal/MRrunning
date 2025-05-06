import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import scipy.stats

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')

# Create directories for results and visualizations
os.makedirs('visualizations', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load and prepare data
print("Loading data...")
df = pd.read_csv('data/Combined_Joined_Data.csv')

# Print original columns for debugging
print("Available columns:")
print(df.columns.tolist())

# Rename columns to match R naming convention
df = df.rename(columns={
    'experiment ID (participant number_condition)': 'ExperimentID',
    'Past exepriences with XR': 'MRExperience_num',
    'participant number': 'ParticipantID',
    'condition': 'Condition_num',
    'run order': 'RunOrder_num',
    'RPE': 'RPE',
    'Satisfaction': 'Satisfaction',
    'Enjoyment': 'Enjoyment',
    'Challenging': 'Challenging',
    'Concentration': 'Concentration',
    'Feeling of presence': 'Presence',
    'Ability to move freely': 'MoveFreely',
    'Fatigue': 'Fatigue',
    'Exhaustion': 'Exhaustion',
    'Sluggish': 'Sluggish',
    'Feeling light weighted': 'LightWeighted',
    'Relaxation and losing oneself in the environment': 'Relaxation',
    'Curiosity and fascination': 'Curiosity',
    'Order in space': 'OrderInSpace',
    'Easy navigation': 'EasyNavigation',
    'Comfortable environment': 'ComfortableEnv',
    'Tense': 'Tense',
    'Anxious': 'Anxious',
    'Healthy': 'Healthy',
    'Angry': 'Angry',
    'Irritated': 'Irritated',
    'Immersed': 'Immersed',
    'Physically present': 'PhysicallyPresent',
    'Movement around and use of objects': 'UseObjects',
    'Ability to do things': 'AbilityToDo',
    'Identification': 'Identification',
    'Greenes of the environment': 'PerceivedGreenness',
    'Sense of belonging': 'Belonging',
    'Gender': 'Gender_num',
    'Average heart rate (bpm)': 'AvgHR',
    'Total distance (km)': 'TotalDistance_km_char',
    'Average speed (km/h)': 'AvgSpeed_kmh_char',
    'Max speed (km/h)': 'MaxSpeed_kmh_char',
    'Average pace (min/km)': 'AvgPace_char',
    'Max pace (min/km)': 'MaxPace_char',
    'Fat percentage of calories(%)': 'FatPercentage',
    'Carbohydrate percentage of calories(%)': 'CarbPercentage',
    'Protein percentage of calories(%)': 'ProteinPercentage',
    'Average cadence (rpm)': 'AvgCadence',
    'Average stride length (cm)': 'AvgStrideLength',
    'Running index': 'RunningIndex',
    'Training load': 'TrainingLoad',
    'Ascent (m)': 'Ascent_m',
    'Descent (m)': 'Descent_m',
    'Average power (W)': 'AvgPower',
    'Max power (W)': 'MaxPower',
    'Duration_seconds': 'Duration_s',
    'Recalculated_Pace_min_per_km': 'Pace_minkm',
    'Recalculated_Speed_km_per_h': 'Speed_kmh',
    'min_heart_rate': 'MinHR',
    'avg_close_to_skin_temp': 'AvgSkinTemp'
})

# Print renamed columns for debugging
print("\nRenamed columns:")
print(df.columns.tolist())

# Convert numeric columns to float
numeric_columns = [
    'RPE', 'Satisfaction', 'Enjoyment', 'Challenging', 'Concentration', 
    'Presence', 'MoveFreely', 'Fatigue', 'Exhaustion', 'Sluggish', 
    'LightWeighted', 'Relaxation', 'Curiosity', 'OrderInSpace', 
    'EasyNavigation', 'ComfortableEnv', 'Tense', 'Anxious', 'Healthy', 
    'Angry', 'Irritated', 'Immersed', 'PhysicallyPresent', 'UseObjects', 
    'AbilityToDo', 'Identification', 'PerceivedGreenness', 'Belonging',
    'Duration_s', 'Speed_kmh', 'MinHR', 'AvgHR', 'VO2max', 'AvgSkinTemp',
    'Age', 'MRExperience_num', 'Gender_num', 'RunOrder_num'
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Create condition variables
df['Condition_num'] = pd.to_numeric(df['Condition_num'], errors='coerce')

# Create main experimental variable (Control vs. Green)
df['IsGreen'] = (df['Condition_num'] > 0).astype(int)

# Create categorical indicator for green type (Shrubs vs. Trees)
df['GreenType'] = df['Condition_num'].map({0: 'Control', 1: 'Shrubs', 2: 'Trees'})

# Calculate composite variables 
presence_items = ['Immersed', 'PhysicallyPresent', 'UseObjects', 'AbilityToDo']
df['Presence_Avg'] = df[presence_items].mean(axis=1)

# Create a Time of Day variable if available
if 'start time' in df.columns:
    df['TimeOfDay'] = pd.to_datetime(df['start time'], errors='coerce').dt.hour
else:
    df['TimeOfDay'] = 12  # Default

# Create composite variables for analysis
prs_items = ['Relaxation', 'Curiosity', 'OrderInSpace', 'EasyNavigation', 'ComfortableEnv']
df['PRS_Avg'] = df[prs_items].mean(axis=1)

pa_items = ['Identification', 'Belonging']
df['PlaceAttach_Avg'] = df[pa_items].mean(axis=1)

positive_affect_items = ['Enjoyment', 'Satisfaction', 'Healthy', 'LightWeighted']
df['PositiveAffect_Avg'] = df[positive_affect_items].mean(axis=1)

negative_affect_items = ['Fatigue', 'Tense', 'Anxious', 'Angry', 'Irritated', 'Sluggish']
df['NegativeAffect_Avg'] = df[negative_affect_items].mean(axis=1)

# Define dependent variables
dep_vars = ['PRS_Avg', 'PlaceAttach_Avg', 'PositiveAffect_Avg', 'NegativeAffect_Avg', 
            'Duration_s', 'RPE', 'Enjoyment', 'Satisfaction', 'Relaxation', 'Healthy', 'LightWeighted', 
            'Fatigue', 'Tense', 'Anxious', 'Angry', 'Irritated', 'Sluggish', 'Concentration', 'MoveFreely', 'Exhaustion', 
            'Presence', 'Immersed', 'PhysicallyPresent', 'UseObjects', 'AbilityToDo',
            'Challenging', 'PerceivedGreenness', 'MinHR', 'Speed_kmh', 'AvgHR']

# Define covariates from explore.py
covariates = ['RunOrder_num', 'Age', 'Gender_num', 'MRExperience_num', 'VO2max', 'AvgSkinTemp', 'TimeOfDay', 'Presence_Avg']

# Remove any rows with missing values in key columns
df = df.dropna(subset=['ParticipantID', 'Condition_num'])

# Storage for results
all_models = {}
model_summaries = {}
decomp_models = {}  # Storage for decomposed models (within/between)
decomp_summaries = {}  # Storage for decomposed model summaries
variance_components = {}

# Check if we have sufficient data for multilevel modeling
participant_counts = df.groupby('ParticipantID').size()
print(f"\nParticipant observation counts:")
print(participant_counts.value_counts())
print(f"Total participants: {len(participant_counts)}")
print(f"Participants with ≥2 observations: {sum(participant_counts >= 2)}")

# Function to run mixed effects model
def run_mixed_effects_model(data, dep_var, covariates_list):
    """
    Run a mixed effects model with participant random intercepts
    to properly estimate within-subject effects while including covariates.
    """
    print(f"\nAnalyzing {dep_var}")
    
    # Drop missing values in the dependent variable
    model_data = data.dropna(subset=[dep_var])
    print(f"  Observations after removing NA: {len(model_data)}")
    
    # Check if we have enough participants with multiple observations
    participant_obs = model_data.groupby('ParticipantID').size()
    participants_with_multiple = sum(participant_obs >= 2)
    print(f"  Participants with multiple observations: {participants_with_multiple}")
    
    if participants_with_multiple < 5:
        print(f"  WARNING: Not enough participants with multiple observations")
        
    # Check for covariates with sufficient non-missing values
    valid_covariates = []
    for cov in covariates_list:
        if cov in model_data.columns and model_data[cov].notna().sum() > len(model_data) * 0.8:
            valid_covariates.append(cov)
    
    # Null model to estimate baseline variance components
    try:
        null_formula = f"{dep_var} ~ 1"
        null_model = smf.mixedlm(null_formula, model_data, groups=model_data["ParticipantID"])
        null_fit = null_model.fit(reml=True)
        
        # Extract variance components
        between_var = null_fit.cov_re.iloc[0, 0]  # Between-participant variance
        within_var = null_fit.scale               # Within-participant variance (residual)
        total_var = between_var + within_var
        icc = between_var / total_var if total_var > 0 else np.nan
        
        print(f"  Variance components (null model):")
        print(f"    Between-participant variance: {between_var:.3f} ({between_var/total_var*100:.1f}%)")
        print(f"    Within-participant variance: {within_var:.3f} ({within_var/total_var*100:.1f}%)")
        print(f"    ICC: {icc:.3f}")
        
        # Store variance components
        variance_components[dep_var] = {
            'null_between_var': between_var,
            'null_within_var': within_var,
            'null_icc': icc
        }
        
    except Exception as e:
        print(f"  Error fitting null model: {str(e)}")
        null_fit = None
    
    # Center all numeric covariates within-subject to separate within and between effects
    # This technique is known as person-mean centering (or within-cluster centering)
    model_data = model_data.copy()
    
    # Identify numeric covariates to center
    numeric_covs = [cov for cov in valid_covariates if cov not in ['ParticipantID', 'ExperimentID', 'Gender_num']]
    
    for cov in numeric_covs:
        # Calculate participant means for each covariate
        cov_means = model_data.groupby('ParticipantID')[cov].transform('mean')
        
        # Create within-subject (centered) version of covariate
        model_data[f'{cov}_within'] = model_data[cov] - cov_means
        
        # Create between-subject version of covariate (participant means)
        model_data[f'{cov}_between'] = cov_means
    
    # Include both non-centered categorical covariates and centered numeric covariates
    # in the model formula
    within_covs_str = ""
    between_covs_str = ""
    categorical_covs_str = ""
    
    for cov in valid_covariates:
        if cov in numeric_covs:
            # For numeric covariates, include both within and between terms
            within_covs_str += f" + {cov}_within"
            between_covs_str += f" + {cov}_between"
        elif cov not in ['ParticipantID', 'ExperimentID']:
            # For categorical covariates, include as is
            categorical_covs_str += f" + {cov}"
    
    # 1. Standard model with treatment coding (IsGreen = 0/1)
    standard_formula = f"{dep_var} ~ IsGreen{within_covs_str}{between_covs_str}{categorical_covs_str}"
    print(f"  Standard formula: {standard_formula}")
    
    standard_fit = None
    try:
        # Fit standard mixed effects model
        standard_model = smf.mixedlm(standard_formula, model_data, groups=model_data["ParticipantID"])
        standard_fit = standard_model.fit(reml=True)
        
        # Extract variance components
        between_var = standard_fit.cov_re.iloc[0, 0]
        within_var = standard_fit.scale
        total_var = between_var + within_var
        icc = between_var / total_var if total_var > 0 else np.nan
        
        # Calculate explained variance
        if null_fit is not None:
            # Proportion of variance explained by fixed effects
            null_total_var = null_fit.cov_re.iloc[0, 0] + null_fit.scale
            explained_var = 1 - (total_var / null_total_var) if null_total_var > 0 else np.nan
        else:
            explained_var = np.nan
        
        # Store variance components from model
        variance_components[dep_var].update({
            'full_between_var': between_var,
            'full_within_var': within_var,
            'full_icc': icc,
            'explained_var': explained_var
        })
        
        # Create summary
        coef_df = pd.DataFrame({
            'Coefficient': standard_fit.params,
            'Std Error': standard_fit.bse,
            'z': standard_fit.tvalues,
            'P>|z|': standard_fit.pvalues,
            'CI 2.5%': standard_fit.conf_int()[0],
            'CI 97.5%': standard_fit.conf_int()[1]
        })
        
        # Add significance indicators
        coef_df['Significance'] = coef_df['P>|z|'].apply(lambda p: 
                                                       '***' if p < 0.001 else
                                                       '**' if p < 0.01 else
                                                       '*' if p < 0.05 else
                                                       '.' if p < 0.1 else '')
        
        # Store model and summary
        all_models[dep_var] = standard_fit
        model_summaries[dep_var] = {
            'coefficients': coef_df,
            'formula': standard_formula,
            'n_obs': len(model_data),
            'n_participants': model_data['ParticipantID'].nunique(),
            'n_with_multiple': participants_with_multiple,
            'log_likelihood': standard_fit.llf,
            'aic': standard_fit.aic,
            'bic': standard_fit.bic
        }
        
        # Print key results
        print(f"  Standard model results:")
        print(f"    IsGreen effect (within-subject): {standard_fit.params['IsGreen']:.3f}, p-value: {standard_fit.pvalues['IsGreen']:.4f}")
        print(f"    Model fit: AIC = {standard_fit.aic:.1f}, BIC = {standard_fit.bic:.1f}")
        print(f"    Within-participant variance explained: {explained_var*100:.1f}%" if not np.isnan(explained_var) else "    Within-participant variance explained: NA")
        
    except Exception as e:
        print(f"  Error fitting standard model: {str(e)}")
    
    # 2. Calculate effect size statistics for the within-subject effect
    # This helps with interpretation alongside the mixed effects model
    complete_participants = []
    for pid, group in model_data.groupby('ParticipantID'):
        if 0 in group['IsGreen'].values and 1 in group['IsGreen'].values:
            complete_participants.append(pid)
    
    print(f"  Participants with both conditions: {len(complete_participants)}")
    
    if len(complete_participants) > 10:
        # Create a dataframe with paired differences for effect size calculation
        diff_data = []
        
        for pid in complete_participants:
            p_data = model_data[model_data['ParticipantID'] == pid]
            if len(p_data) == 2:  # Exactly one control and one green observation
                control_val = p_data[p_data['IsGreen'] == 0][dep_var].values[0]
                green_val = p_data[p_data['IsGreen'] == 1][dep_var].values[0]
                
                # Store within-subject difference
                diff = green_val - control_val
                diff_data.append({
                    'ParticipantID': pid,
                    'Difference': diff,
                })
        
        # Convert to dataframe
        if diff_data:
            within_df = pd.DataFrame(diff_data)
            
            # Calculate Cohen's d (standardized effect size)
            mean_diff = within_df['Difference'].mean()
            sd_diff = within_df['Difference'].std()
            cohens_d = mean_diff / sd_diff if sd_diff > 0 else np.nan
            se_diff = sd_diff / np.sqrt(len(within_df))
            
            # Store effect size results
            effect_size_stats = {
                'mean_diff': mean_diff,
                'sd_diff': sd_diff,
                'se_diff': se_diff,
                'cohens_d': cohens_d,
                'n': len(within_df)
            }
            
            decomp_models[dep_var] = effect_size_stats
            
            # Print effect size
            sig = "***" if standard_fit and standard_fit.pvalues['IsGreen'] < 0.001 else "**" if standard_fit and standard_fit.pvalues['IsGreen'] < 0.01 else "*" if standard_fit and standard_fit.pvalues['IsGreen'] < 0.05 else "." if standard_fit and standard_fit.pvalues['IsGreen'] < 0.1 else ""
            print(f"  Effect size statistics:")
            print(f"    Raw mean difference: {mean_diff:.3f} {sig}")
            print(f"    Cohen's d: {cohens_d:.3f}")
            
            # Store summary statistics
            decomp_summaries[dep_var] = {
                'mean_diff': mean_diff,
                'sd_diff': sd_diff,
                'se_diff': se_diff,
                'cohens_d': cohens_d,
                'n': len(within_df)
            }

    return standard_fit

# Run mixed effects models for each dependent variable
for dep_var in dep_vars:
    run_mixed_effects_model(df, dep_var, covariates)

# Create visualization functions
def create_within_subject_plot(data, dep_var):
    """Create a visualization showing within-subject effects"""
    # Check if we have sufficient data
    participant_counts = data.groupby('ParticipantID').size()
    participants_with_multiple = participant_counts[participant_counts >= 2].index
    
    if len(participants_with_multiple) < 5:
        print(f"  Not enough participants with multiple observations for {dep_var} plot")
        return
    
    # Filter to participants with multiple observations
    plot_data = data[data['ParticipantID'].isin(participants_with_multiple)]
    plot_data = plot_data.dropna(subset=[dep_var])
    
    # Create figure
    plt.figure(figsize=(10, 7))
    
    # Extract control and green data for each participant
    lines_data = []
    for pid in participants_with_multiple:
        p_data = plot_data[plot_data['ParticipantID'] == pid]
        if len(p_data) >= 2 and 0 in p_data['IsGreen'].values and 1 in p_data['IsGreen'].values:
            ctrl_val = p_data[p_data['IsGreen'] == 0][dep_var].values[0]
            green_val = p_data[p_data['IsGreen'] == 1][dep_var].values[0]
            lines_data.append((ctrl_val, green_val))
            # Plot individual participant lines
            plt.plot([0, 1], [ctrl_val, green_val], 'o-', color='gray', alpha=0.4, linewidth=1)
    
    # If we have any valid pairs
    if lines_data:
        # Convert to numpy array for easy calculations
        lines_array = np.array(lines_data)
        # Calculate means for control and green
        ctrl_mean = lines_array[:, 0].mean()
        green_mean = lines_array[:, 1].mean()
        # Calculate standard errors
        ctrl_se = lines_array[:, 0].std() / np.sqrt(len(lines_data))
        green_se = lines_array[:, 1].std() / np.sqrt(len(lines_data))
        
        # Plot the mean line
        plt.plot([0, 1], [ctrl_mean, green_mean], 'o-', color='red', linewidth=3, 
                label=f'Mean (n={len(lines_data)})')
        # Add error bars
        plt.errorbar([0, 1], [ctrl_mean, green_mean], yerr=[ctrl_se, green_se], 
                    fmt='none', color='red', capsize=5)
        
        # Add the mean values as text
        plt.text(0, ctrl_mean, f"{ctrl_mean:.2f}±{ctrl_se:.2f}", ha='right', va='bottom')
        plt.text(1, green_mean, f"{green_mean:.2f}±{green_se:.2f}", ha='left', va='bottom')
        
        # Get the standard model and effect size stats
        standard_model = all_models.get(dep_var)
        effect_stats = decomp_models.get(dep_var)
        
        # Build title with model effects
        title_parts = [f"Within-subject Effect of Green Environment on {dep_var}"]
        
        # Add mixed model effect if available
        if standard_model is not None and 'IsGreen' in standard_model.params:
            effect = standard_model.params['IsGreen']
            p_value = standard_model.pvalues['IsGreen']
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "." if p_value < 0.1 else ""
            title_parts.append(f"Mixed Model Effect: {effect:.3f} {sig} (p={p_value:.4f})")
        
        # Add effect size if available
        if effect_stats is not None and 'cohens_d' in effect_stats:
            mean_diff = effect_stats['mean_diff']
            d = effect_stats['cohens_d']
            title_parts.append(f"Cohen's d = {d:.2f}, Raw Difference = {mean_diff:.3f}")
        
        plt.title('\n'.join(title_parts))
    
    # Set axis labels and grid
    plt.xticks([0, 1], ['Control', 'Green'], fontsize=12)
    plt.ylabel(dep_var, fontsize=12)
    plt.grid(alpha=0.3)
    
    # Save the figure
    os.makedirs(f'visualizations/{dep_var}', exist_ok=True)
    plt.savefig(f'visualizations/{dep_var}/within_subject_effect.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_variance_components_plot(data, dep_var):
    """Create a visualization of variance components"""
    if dep_var not in variance_components:
        return
    
    vc = variance_components[dep_var]
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Data for stacked bar chart
    between_var = vc.get('full_between_var', 0)
    within_var = vc.get('full_within_var', 0)
    total_var = between_var + within_var
    
    # Calculate percentages
    between_pct = between_var / total_var * 100 if total_var > 0 else 0
    within_pct = within_var / total_var * 100 if total_var > 0 else 0
    
    # Create stacked bar
    plt.bar([0], [between_pct], color='blue', alpha=0.7, label=f'Between-participant ({between_pct:.1f}%)')
    plt.bar([0], [within_pct], bottom=[between_pct], color='orange', alpha=0.7, 
           label=f'Within-participant ({within_pct:.1f}%)')
    
    # Add the ICC
    icc = vc.get('full_icc', np.nan)
    plt.text(0, 105, f"ICC = {icc:.3f}", ha='center', fontsize=12, fontweight='bold')
    
    # Add explained variance if available
    explained_var = vc.get('explained_var', np.nan)
    if not np.isnan(explained_var):
        plt.text(0, 115, f"Var. explained = {explained_var*100:.1f}%", ha='center', fontsize=10)
    
    # Set labels
    plt.ylabel('Percentage of Total Variance (%)')
    plt.title(f'Variance Components for {dep_var}')
    plt.xticks([0], ['Variance Partition'])
    plt.ylim(0, 120)  # Make room for the text
    plt.legend(loc='upper right')
    
    # Save the figure
    os.makedirs(f'visualizations/{dep_var}', exist_ok=True)
    plt.savefig(f'visualizations/{dep_var}/variance_components.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create visualizations for each dependent variable
for dep_var in dep_vars:
    if dep_var in all_models:
        print(f"Creating visualizations for {dep_var}")
        create_within_subject_plot(df, dep_var)
        create_variance_components_plot(df, dep_var)

# Create summary Excel file
with pd.ExcelWriter('results/mixed_effects_results.xlsx', engine='openpyxl') as writer:
    
    # Create a summary sheet for all models
    summary_rows = []
    for dep_var in dep_vars:
        if dep_var in all_models:
            # Get model results
            standard_model = all_models[dep_var]
            effect_stats = decomp_models.get(dep_var)
            vc = variance_components.get(dep_var, {})
            
            # Initialize row with basic information
            row = {
                'Dependent Variable': dep_var,
                'Observations': model_summaries[dep_var]['n_obs'],
                'Participants': model_summaries[dep_var]['n_participants'],
                'ICC': vc.get('full_icc', np.nan),
                'Between-subject Var': vc.get('full_between_var', np.nan),
                'Within-subject Var': vc.get('full_within_var', np.nan),
                'Var Explained (%)': vc.get('explained_var', np.nan) * 100 if 'explained_var' in vc else np.nan
            }
            
            # Add mixed model results (these are within-subject effects with covariates)
            if 'IsGreen' in standard_model.params:
                effect_size = standard_model.params['IsGreen']
                p_value = standard_model.pvalues['IsGreen']
                ci_lower = standard_model.conf_int().loc['IsGreen', 0]
                ci_upper = standard_model.conf_int().loc['IsGreen', 1]
                
                # Determine significance indicator
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "." if p_value < 0.1 else ""
                
                row.update({
                    'IsGreen Effect': effect_size,
                    'IsGreen P-value': p_value,
                    'IsGreen Sig': sig,
                    'IsGreen 95% CI Lower': ci_lower,
                    'IsGreen 95% CI Upper': ci_upper
                })
            
            # Add effect size statistics if available
            if effect_stats is not None and 'cohens_d' in effect_stats:
                mean_diff = effect_stats['mean_diff']
                sd_diff = effect_stats['sd_diff']
                se_diff = effect_stats['se_diff']
                cohens_d = effect_stats['cohens_d']
                n = effect_stats['n']
                
                # Calculate t-value and p-value
                t_value = mean_diff / se_diff if se_diff > 0 else np.nan
                p_value = 2 * (1 - scipy.stats.t.cdf(abs(t_value), n-1)) if not np.isnan(t_value) else np.nan
                
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "." if p_value < 0.1 else ""
                
                row.update({
                    'Raw Mean Diff': mean_diff,
                    'SD of Diff': sd_diff,
                    'SE of Diff': se_diff,
                    'Cohens d': cohens_d,
                    't-value': t_value,
                    'Paired P-value': p_value,
                    'Paired Sig': sig
                })
            
            # Add all covariate effects from standard model to the row
            for cov in standard_model.params.index:
                if cov != 'Intercept' and cov != 'IsGreen' and not cov.endswith('_between'):
                    effect_size = standard_model.params[cov]
                    p_value = standard_model.pvalues[cov]
                    
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "." if p_value < 0.1 else ""
                    
                    row.update({
                        f'{cov} Effect': effect_size,
                        f'{cov} P-value': p_value,
                        f'{cov} Sig': sig
                    })
            
            summary_rows.append(row)
    
    # Create DataFrame and sort by significance of IsGreen effect
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if 'IsGreen P-value' in summary_df.columns:
            summary_df = summary_df.sort_values(['IsGreen P-value'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # Create detailed sheets for each model
    for dep_var, model in all_models.items():
        if model is not None and dep_var in model_summaries:
            # Clean sheet name
            safe_name = f"{dep_var[:20]}_model"
            
            # Get coefficient table
            coef_df = model_summaries[dep_var]['coefficients']
            
            # Write coefficients
            coef_df.to_excel(writer, sheet_name=safe_name, index=True, startrow=0)
            
            # Create diagnostics DataFrame
            diagnostics = pd.DataFrame({
                'Metric': [
                    'Dependent Variable',
                    'Formula',
                    'Observations',
                    'Participants',
                    'ICC',
                    'Between-participant Variance',
                    'Within-participant Variance',
                    'Variance Explained (%)',
                    'Log-Likelihood',
                    'AIC',
                    'BIC'
                ],
                'Value': [
                    dep_var,
                    model_summaries[dep_var]['formula'],
                    model_summaries[dep_var]['n_obs'],
                    model_summaries[dep_var]['n_participants'],
                    variance_components.get(dep_var, {}).get('full_icc', np.nan),
                    variance_components.get(dep_var, {}).get('full_between_var', np.nan),
                    variance_components.get(dep_var, {}).get('full_within_var', np.nan),
                    variance_components.get(dep_var, {}).get('explained_var', np.nan) * 100 if dep_var in variance_components and 'explained_var' in variance_components[dep_var] else np.nan,
                    model_summaries[dep_var].get('log_likelihood', np.nan),
                    model_summaries[dep_var].get('aic', np.nan),
                    model_summaries[dep_var].get('bic', np.nan)
                ]
            })
            
            # Write diagnostics below coefficients
            diagnostics.to_excel(writer, sheet_name=safe_name, index=False, startrow=len(coef_df) + 3)
    
    # Create detailed sheets for effect size statistics
    for dep_var, effect_stats in decomp_models.items():
        if effect_stats is not None:
            # Clean sheet name
            safe_name = f"{dep_var[:20]}_effect"
            
            # Calculate t-value and p-value
            mean_diff = effect_stats['mean_diff']
            se_diff = effect_stats['se_diff']
            n = effect_stats['n']
            t_value = mean_diff / se_diff if se_diff > 0 else np.nan
            p_value = 2 * (1 - scipy.stats.t.cdf(abs(t_value), n-1)) if not np.isnan(t_value) else np.nan
            
            # Create effect size statistics table
            effect_df = pd.DataFrame({
                'Metric': [
                    'Dependent Variable',
                    'Mean Difference (Green - Control)',
                    'Standard Deviation of Differences',
                    'Standard Error of Mean Difference',
                    't-value',
                    'p-value',
                    'Degrees of Freedom',
                    'Cohen\'s d',
                    'n'
                ],
                'Value': [
                    dep_var,
                    mean_diff,
                    effect_stats['sd_diff'],
                    se_diff,
                    t_value,
                    p_value,
                    n-1,
                    effect_stats['cohens_d'],
                    n
                ]
            })
            
            # Write effect size statistics
            effect_df.to_excel(writer, sheet_name=safe_name, index=False, startrow=0)

print("\nAnalysis complete. Results saved to results/mixed_effects_results.xlsx")
print("Visualizations saved to the 'visualizations' directory") 