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
covariates = ['RunOrder_num', 'Age', 'Gender_num', 'VO2max', 'AvgSkinTemp', 'Presence_Avg']

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
    Uses a simplified approach to avoid singular matrix errors.
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
    
    # Determine which covariates to use based on available data
    # Only use a small set of key covariates to avoid overfitting
    valid_covariates = []
    
    # Check for sufficient non-missing values (at least 80% present)
    for cov in covariates_list:
        if cov in model_data.columns and model_data[cov].notna().sum() > len(model_data) * 0.8:
            # Don't include composite variables if we have their components
            if cov == 'Presence_Avg' and 'Presence' in dep_vars:
                continue
            valid_covariates.append(cov)
    
    # For very limited models, we'll only use a few key covariates
    # Keep these to maximum 3-4 covariates for stability
    key_covariates = ['RunOrder_num', 'Gender_num']
    
    # Add one physiological variable if available
    if 'VO2max' in valid_covariates:
        key_covariates.append('VO2max')
    elif 'AvgHR' in valid_covariates and dep_var != 'AvgHR':
        key_covariates.append('AvgHR')
        
    # Ensure we're not overfitting
    valid_covariates = [cov for cov in key_covariates if cov in valid_covariates]
    print(f"  Using covariates: {valid_covariates}")
    
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
    
    # Create copy of model data for analysis
    model_data = model_data.copy()
    
    # Create IsGreen variable based on Condition_num
    if 'IsGreen' not in model_data.columns:
        model_data['IsGreen'] = (model_data['Condition_num'] > 0).astype(int)
    
    # For categorical covariates, we'll use them as-is
    categorical_covs = [cov for cov in valid_covariates if cov in ['Gender_num']]
    
    # For numeric covariates, we'll center them for within-subject effects
    # But we won't create separate between-subject terms to reduce parameters
    numeric_covs = [cov for cov in valid_covariates if cov not in categorical_covs and cov != 'ParticipantID']
    
    for cov in numeric_covs:
        # Group-mean centering - subtract the participant's mean
        # This isolates the within-subject effect
        model_data[f'{cov}_c'] = model_data.groupby('ParticipantID')[cov].transform(
            lambda x: x - x.mean()
        )
    
    # Build formula components
    if numeric_covs:
        # Use centered versions for all numeric covariates
        numeric_terms = " + ".join([f"{cov}_c" for cov in numeric_covs])
        if numeric_terms:
            numeric_terms = " + " + numeric_terms
    else:
        numeric_terms = ""
    
    if categorical_covs:
        categorical_terms = " + " + " + ".join(categorical_covs)
    else:
        categorical_terms = ""
    
    # For GreenType, include as a main effect and interaction
    if 'GreenType' in model_data.columns:
        # Create a binary indicator for Trees vs Shrubs
        # Only among green conditions, to avoid confounding with the IsGreen effect
        green_rows = model_data['IsGreen'] == 1
        model_data['IsTree'] = 0
        model_data.loc[green_rows & (model_data['GreenType'] == 'Trees'), 'IsTree'] = 1
        
        # Include interaction with IsGreen
        green_type_terms = " + IsTree + IsGreen:IsTree"
    else:
        green_type_terms = ""
    
    # Create simplified formula
    mixed_formula = f"{dep_var} ~ IsGreen{numeric_terms}{categorical_terms}{green_type_terms}"
    print(f"  Mixed model formula: {mixed_formula}")
    
    # First try with standard mixed model
    try:
        # Fit mixed effects model with standard approach
        mixed_model = smf.mixedlm(mixed_formula, model_data, groups=model_data["ParticipantID"])
        mixed_fit = mixed_model.fit(reml=True)
        
        # Success! Extract results
        print(f"  Mixed model fitted successfully")
        
        # Extract variance components
        between_var = mixed_fit.cov_re.iloc[0, 0]
        within_var = mixed_fit.scale
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
            'Coefficient': mixed_fit.params,
            'Std Error': mixed_fit.bse,
            'z': mixed_fit.tvalues,
            'P>|z|': mixed_fit.pvalues,
            'CI 2.5%': mixed_fit.conf_int()[0],
            'CI 97.5%': mixed_fit.conf_int()[1]
        })
        
        # Add significance indicators
        coef_df['Significance'] = coef_df['P>|z|'].apply(lambda p: 
                                                       '***' if p < 0.001 else
                                                       '**' if p < 0.01 else
                                                       '*' if p < 0.05 else
                                                       '.' if p < 0.1 else '')
        
        # Categorize parameters as within or between effects
        coef_df['Effect_Type'] = coef_df.index.map(
            lambda x: 'Within-Subject' if x == 'IsGreen' or x.endswith('_c') or 'IsGreen:' in x 
                    else 'Between-Subject' if x == 'IsTree' or x in categorical_covs
                    else 'Intercept' if x == 'Intercept' 
                    else 'Other'
        )
        
        # Store model and summary
        all_models[dep_var] = mixed_fit
        model_summaries[dep_var] = {
            'coefficients': coef_df,
            'formula': mixed_formula,
            'n_obs': len(model_data),
            'n_participants': model_data['ParticipantID'].nunique(),
            'n_with_multiple': participants_with_multiple,
            'log_likelihood': mixed_fit.llf,
            'aic': mixed_fit.aic,
            'bic': mixed_fit.bic
        }
        
        # Print key results
        print(f"  Mixed model results:")
        print(f"    IsGreen effect (within-subject): {mixed_fit.params['IsGreen']:.3f}, p-value: {mixed_fit.pvalues['IsGreen']:.4f}")
        
        # Print interaction effects if they exist
        if 'IsGreen:IsTree' in mixed_fit.params:
            print(f"    IsGreen x Trees interaction: {mixed_fit.params['IsGreen:IsTree']:.3f}, p-value: {mixed_fit.pvalues['IsGreen:IsTree']:.4f}")
        
        print(f"    Model fit: AIC = {mixed_fit.aic:.1f}, BIC = {mixed_fit.bic:.1f}")
        print(f"    Within-participant variance explained: {explained_var*100:.1f}%" if not np.isnan(explained_var) else "    Within-participant variance explained: NA")
        
    except Exception as e:
        # If standard mixed model fails, try robust approach
        print(f"  Error fitting standard mixed model: {str(e)}")
        print(f"  Trying simplified model...")
        
        # Further simplify formula - remove interactions and use just IsGreen
        simplified_formula = f"{dep_var} ~ IsGreen"
        
        try:
            # Try with just IsGreen
            mixed_model = smf.mixedlm(simplified_formula, model_data, groups=model_data["ParticipantID"])
            mixed_fit = mixed_model.fit(reml=True)
            
            print(f"  Simplified mixed model fitted successfully")
            
            # Extract variance components
            between_var = mixed_fit.cov_re.iloc[0, 0]
            within_var = mixed_fit.scale
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
                'Coefficient': mixed_fit.params,
                'Std Error': mixed_fit.bse,
                'z': mixed_fit.tvalues,
                'P>|z|': mixed_fit.pvalues,
                'CI 2.5%': mixed_fit.conf_int()[0],
                'CI 97.5%': mixed_fit.conf_int()[1]
            })
            
            # Add significance indicators
            coef_df['Significance'] = coef_df['P>|z|'].apply(lambda p: 
                                                       '***' if p < 0.001 else
                                                       '**' if p < 0.01 else
                                                       '*' if p < 0.05 else
                                                       '.' if p < 0.1 else '')
            
            # Categorize parameters as within or between effects
            coef_df['Effect_Type'] = coef_df.index.map(
                lambda x: 'Within-Subject' if x == 'IsGreen' 
                        else 'Intercept' if x == 'Intercept' 
                        else 'Other'
            )
            
            # Store model and summary
            all_models[dep_var] = mixed_fit
            model_summaries[dep_var] = {
                'coefficients': coef_df,
                'formula': simplified_formula,
                'n_obs': len(model_data),
                'n_participants': model_data['ParticipantID'].nunique(),
                'n_with_multiple': participants_with_multiple,
                'log_likelihood': mixed_fit.llf,
                'aic': mixed_fit.aic,
                'bic': mixed_fit.bic
            }
            
            # Print key results
            print(f"  Simplified mixed model results:")
            print(f"    IsGreen effect (within-subject): {mixed_fit.params['IsGreen']:.3f}, p-value: {mixed_fit.pvalues['IsGreen']:.4f}")
            print(f"    Model fit: AIC = {mixed_fit.aic:.1f}, BIC = {mixed_fit.bic:.1f}")
            print(f"    Within-participant variance explained: {explained_var*100:.1f}%" if not np.isnan(explained_var) else "    Within-participant variance explained: NA")
            
        except Exception as e:
            print(f"  Error fitting simplified mixed model: {str(e)}")
            mixed_fit = None
    
    # 2. Fit separate between-subject model (using participant means)
    try:
        # Create dataframe of participant means
        between_data = model_data.groupby('ParticipantID').agg({
            dep_var: 'mean',
            'IsGreen': 'mean'  # This will be proportion of green trials (usually 0.5 with 2 observations)
        })
        
        # Add green type if applicable
        if 'GreenType' in model_data.columns:
            # Most common green type by participant
            between_data['GreenType'] = model_data.groupby('ParticipantID')['GreenType'].apply(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else None
            )
            
            # Add IsTree indicator
            between_data['IsTree'] = between_data['GreenType'].map(lambda x: 1 if x == 'Trees' else 0)
        
        # Add participant-level covariates (means of variables)
        for cov in valid_covariates:
            if cov not in ['ParticipantID', 'ExperimentID', 'IsGreen']:
                between_data[cov] = model_data.groupby('ParticipantID')[cov].mean()
        
        # Create formula for between-subject model
        # Keep it simple with a minimal set of covariates
        between_formula = f"{dep_var} ~ IsGreen"
        
        # Add GreenType interaction if available
        if 'GreenType' in between_data.columns:
            between_formula += " * IsTree"
        
        # Add a few key covariates only
        covariate_terms = [cov for cov in valid_covariates if cov != 'ParticipantID']
        if covariate_terms:
            between_formula += " + " + " + ".join(covariate_terms)
        
        print(f"  Between-subject formula: {between_formula}")
        
        # Fit between-subject model
        between_model = smf.ols(between_formula, between_data).fit()
        
        # Create summary
        between_coef_df = pd.DataFrame({
            'Coefficient': between_model.params,
            'Std Error': between_model.bse,
            't': between_model.tvalues,
            'P>|t|': between_model.pvalues,
            'CI 2.5%': between_model.conf_int()[0],
            'CI 97.5%': between_model.conf_int()[1]
        })
        
        # Add significance indicators
        between_coef_df['Significance'] = between_coef_df['P>|t|'].apply(lambda p: 
                                                                    '***' if p < 0.001 else
                                                                    '**' if p < 0.01 else
                                                                    '*' if p < 0.05 else
                                                                    '.' if p < 0.1 else '')
        
        # Store between-subject model and summary
        decomp_models[f"{dep_var}_between"] = between_model
        decomp_summaries[f"{dep_var}_between"] = {
            'coefficients': between_coef_df,
            'formula': between_formula,
            'n_obs': len(between_data),
            'r_squared': between_model.rsquared,
            'r_squared_adj': between_model.rsquared_adj,
            'f_statistic': between_model.fvalue,
            'f_pvalue': between_model.f_pvalue
        }
        
        # Print key results
        print(f"  Between-subject model results:")
        print(f"    IsGreen proportion effect: {between_model.params.get('IsGreen', np.nan):.3f}, p-value: {between_model.pvalues.get('IsGreen', np.nan):.4f}")
        if 'IsGreen:IsTree' in between_model.params:
            print(f"    IsGreen x Trees interaction: {between_model.params['IsGreen:IsTree']:.3f}, p-value: {between_model.pvalues['IsGreen:IsTree']:.4f}")
        print(f"    Model fit: R² = {between_model.rsquared:.3f}, Adj. R² = {between_model.rsquared_adj:.3f}")
        
    except Exception as e:
        print(f"  Error fitting between-subject model: {str(e)}")
    
    return mixed_fit

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
        
        # Get the mixed effects model
        model = all_models.get(dep_var)
        
        # Build title with model effects
        title_parts = [f"Within-subject Effect of Green Environment on {dep_var}"]
        
        # Add mixed model effect if available
        if model is not None and 'IsGreen' in model.params:
            effect = model.params['IsGreen']
            p_value = model.pvalues['IsGreen']
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "." if p_value < 0.1 else ""
            title_parts.append(f"Mixed Model Effect: {effect:.3f} {sig} (p={p_value:.4f})")
        
            # Add raw difference for context (but not standardized)
            raw_diff = green_mean - ctrl_mean
            title_parts.append(f"Raw Mean Difference: {raw_diff:.3f}")
        
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
    
    # Add a test sheet to ensure we have at least one valid sheet
    pd.DataFrame({'Test': ['This is a test sheet']}).to_excel(writer, sheet_name='Test', index=False)
    
    # Create a summary sheet for all models
    summary_rows = []
    for dep_var in dep_vars:
        if dep_var in all_models:
            # Get model results
            standard_model = all_models[dep_var]
            between_model = decomp_models.get(f"{dep_var}_between")
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
            
            # Add interaction effects from mixed model
            for param in standard_model.params.index:
                if 'IsGreen:C(GreenType' in param:
                    effect_size = standard_model.params[param]
                    p_value = standard_model.pvalues[param]
                    ci_lower = standard_model.conf_int().loc[param, 0]
                    ci_upper = standard_model.conf_int().loc[param, 1]
                    
                    # Clean parameter name
                    clean_param = param.replace('IsGreen:C(GreenType)', 'IsGreen x GreenType')
                    
                    # Determine significance indicator
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "." if p_value < 0.1 else ""
                    
                    row.update({
                        f'{clean_param} Effect': effect_size,
                        f'{clean_param} P-value': p_value,
                        f'{clean_param} Sig': sig
                    })
            
            # Add between-subject model results if available
            if between_model is not None:
                if hasattr(between_model, 'params') and 'IsGreen' in between_model.params:
                    between_effect = between_model.params['IsGreen']
                    between_p = between_model.pvalues['IsGreen']
                    between_sig = "***" if between_p < 0.001 else "**" if between_p < 0.01 else "*" if between_p < 0.05 else "." if between_p < 0.1 else ""
                    
                    row.update({
                        'Between IsGreen Effect': between_effect,
                        'Between IsGreen P-value': between_p,
                        'Between IsGreen Sig': between_sig
                    })
                
                # Add interaction term from between-subject model
                if hasattr(between_model, 'params') and 'IsGreen:GreenType' in between_model.params:
                    interact_effect = between_model.params['IsGreen:GreenType']
                    interact_p = between_model.pvalues['IsGreen:GreenType']
                    interact_sig = "***" if interact_p < 0.001 else "**" if interact_p < 0.01 else "*" if interact_p < 0.05 else "." if interact_p < 0.1 else ""
                    
                    row.update({
                        'Between IsGreen x GreenType Effect': interact_effect,
                        'Between IsGreen x GreenType P-value': interact_p,
                        'Between IsGreen x GreenType Sig': interact_sig
                    })
                
                # Add model fit statistics
                if hasattr(between_model, 'rsquared'):
                    row.update({
                            'Between R-squared': between_model.rsquared,
                            'Between Adj R-squared': between_model.rsquared_adj
                    })
            
            # Add all covariate effects from standard model to the row
            for cov in standard_model.params.index:
                if cov != 'Intercept' and cov != 'IsGreen' and not 'IsGreen:C(GreenType' in cov:
                    # Skip GreenType main effects for clarity
                    if cov.startswith('C(GreenType'):
                        continue
                        
                    effect_type = 'Within' if cov.endswith('_c') else 'Between' if cov.endswith('_c') else 'Other'
                    cov_clean = cov.replace('_c', '')
                    
                    effect_size = standard_model.params[cov]
                    p_value = standard_model.pvalues[cov]
                    
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "." if p_value < 0.1 else ""
                    
                    row.update({
                        f'{effect_type} {cov_clean} Effect': effect_size,
                        f'{effect_type} {cov_clean} P-value': p_value,
                        f'{effect_type} {cov_clean} Sig': sig
                    })
            
            summary_rows.append(row)
    
    # Create DataFrame and sort by significance of IsGreen effect
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if 'IsGreen P-value' in summary_df.columns:
            summary_df = summary_df.sort_values(['IsGreen P-value'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # Create one sheet per dependent variable with all model results
    for i, dep_var in enumerate(dep_vars):
        if dep_var in all_models:
            # Use simple sheet names to avoid any issues
            safe_name = f"Var{i+1}"
            
            # Create a combined results dataframe for this variable
            results_data = []
            
            # Add header/title
            results_data.append(["Results for " + dep_var, "", "", "", ""])
            results_data.append(["", "", "", "", ""])  # Empty row for spacing
            
            # Add variance components
            vc = variance_components.get(dep_var, {})
            results_data.append(["Variance Components", "", "", "", ""])
            results_data.append(["ICC", vc.get('full_icc', np.nan), "", "", ""])
            results_data.append(["Between-subject Variance", vc.get('full_between_var', np.nan), "", "", ""])
            results_data.append(["Within-subject Variance", vc.get('full_within_var', np.nan), "", "", ""])
            results_data.append(["Variance Explained (%)", vc.get('explained_var', np.nan) * 100 if 'explained_var' in vc else np.nan, "", "", ""])
            results_data.append(["", "", "", "", ""])  # Empty row for spacing
            
            # Add mixed model results
            mixed_model = all_models[dep_var]
            results_data.append(["Mixed Effects Model Results", "", "", "", ""])
            results_data.append(["Formula", model_summaries[dep_var]['formula'], "", "", ""])
            results_data.append(["Observations", model_summaries[dep_var]['n_obs'], "", "", ""])
            results_data.append(["Participants", model_summaries[dep_var]['n_participants'], "", "", ""])
            results_data.append(["Log-Likelihood", model_summaries[dep_var].get('log_likelihood', np.nan), "", "", ""])
            results_data.append(["AIC", model_summaries[dep_var].get('aic', np.nan), "", "", ""])
            results_data.append(["BIC", model_summaries[dep_var].get('bic', np.nan), "", "", ""])
            results_data.append(["", "", "", "", ""])  # Empty row for spacing
            
            # Add mixed model coefficients (sorted by effect type)
            coef_df = model_summaries[dep_var]['coefficients'].copy()
            if 'Effect_Type' in coef_df.columns:
                effect_type_order = {'Intercept': 0, 'Within-Subject': 1, 'Between-Subject': 2, 'Other': 3}
                coef_df['sort_order'] = coef_df['Effect_Type'].map(effect_type_order)
                coef_df = coef_df.sort_values('sort_order')
                coef_df = coef_df.drop(columns=['sort_order'])
            
            # Convert coefficients to rows
            results_data.append(["Mixed Model Coefficients", "", "", "", ""])
            results_data.append(["Parameter", "Coefficient", "Std Error", "P>|z|", "Significance"])
            
            for idx, row in coef_df.iterrows():
                results_data.append([
                    idx, 
                    row['Coefficient'], 
                    row['Std Error'],
                    row['P>|z|'],
                    row['Significance']
                ])
            
            results_data.append(["", "", "", "", ""])  # Empty row for spacing
            
            # Add between-subject model results if available
            between_model = decomp_models.get(f"{dep_var}_between")
            if between_model is not None and hasattr(between_model, 'params'):
                between_summary = decomp_summaries.get(f"{dep_var}_between")
                
                results_data.append(["Between-Subject Model Results", "", "", "", ""])
                if between_summary:
                    results_data.append(["Formula", between_summary.get('formula', ""), "", "", ""])
                    results_data.append(["Observations", between_summary.get('n_obs', np.nan), "", "", ""])
                    results_data.append(["R-squared", between_summary.get('r_squared', np.nan), "", "", ""])
                    results_data.append(["Adjusted R-squared", between_summary.get('r_squared_adj', np.nan), "", "", ""])
                    results_data.append(["F-statistic", between_summary.get('f_statistic', np.nan), "", "", ""])
                    results_data.append(["F p-value", between_summary.get('f_pvalue', np.nan), "", "", ""])
                    results_data.append(["", "", "", "", ""])  # Empty row for spacing
                
                # If we have coefficients
                if between_summary and 'coefficients' in between_summary:
                    between_coef_df = between_summary['coefficients']
                    
                    # Convert coefficients to rows
                    results_data.append(["Between-Subject Model Coefficients", "", "", "", ""])
                    results_data.append(["Parameter", "Coefficient", "Std Error", "P>|t|", "Significance"])
                    
                    for idx, row in between_coef_df.iterrows():
                        results_data.append([
                            idx, 
                            row['Coefficient'], 
                            row['Std Error'],
                            row['P>|t|'] if 'P>|t|' in row else row.get('P>|z|', np.nan),
                            row['Significance']
                        ])
            
            # Convert to DataFrame and write to Excel
            results_df = pd.DataFrame(results_data)
            results_df.to_excel(writer, sheet_name=safe_name, header=False, index=False)
    
    # Add a mapping sheet that shows which variable is in which sheet
    var_mapping = []
    for i, dep_var in enumerate(dep_vars):
        if dep_var in all_models:
            var_mapping.append({'Sheet': f"Var{i+1}", 'Variable': dep_var})
    
    if var_mapping:
        pd.DataFrame(var_mapping).to_excel(writer, sheet_name='VariableMap', index=False)

print("\nAnalysis complete. Results saved to results/mixed_effects_results.xlsx")
print("Visualizations saved to the 'visualizations' directory") 