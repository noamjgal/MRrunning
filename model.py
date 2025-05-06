import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import warnings
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter
from scipy import stats

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')

# Create directory for results
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

# Create main experimental variables
df['IsGreen'] = (df['Condition_num'] > 0).astype(int)  # 1 if any green, 0 if control
df['IsShrub'] = (df['Condition_num'] == 1).astype(int)  # 1 if shrub, 0 otherwise
df['IsTree'] = (df['Condition_num'] == 2).astype(int)   # 1 if tree, 0 otherwise

# Create interaction variables
df['Green_Shrub'] = df['IsGreen'] * df['IsShrub']  # 1 if shrub, 0 otherwise
df['Green_Tree'] = df['IsGreen'] * df['IsTree']    # 1 if tree, 0 otherwise

# Print condition counts
print("\nCondition counts:")
print(df['Condition_num'].value_counts().sort_index())
print("\nIsGreen counts:")
print(df['IsGreen'].value_counts())
print("\nGreen type counts:")
print("Shrubs:", df['IsShrub'].sum())
print("Trees:", df['IsTree'].sum())

# Calculate composite variables 
presence_items = ['Immersed', 'PhysicallyPresent', 'UseObjects', 'AbilityToDo']
df['Presence_Avg'] = df[presence_items].mean(axis=1)

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

# Define covariates - exact list as specified
covariates = ['RunOrder_num', 'Age', 'Gender_num', 'VO2max', 'AvgSkinTemp', 'Presence_Avg']

# Remove any rows with missing values in key columns
df_original_count = len(df)
df = df.dropna(subset=['ParticipantID', 'Condition_num'])
df_filtered_count = len(df)

print(f"\nFiltered data: {df_filtered_count} observations (removed {df_original_count - df_filtered_count} rows with missing ParticipantID or Condition_num)")

# Check if we have sufficient data for analysis
participant_counts = df.groupby('ParticipantID').size()
print(f"\nParticipant observation counts:")
print(participant_counts.value_counts())
print(f"Total participants: {len(participant_counts)}")
print(f"Participants with ≥2 observations: {sum(participant_counts >= 2)}")

# Storage for results
all_models = {}
model_summaries = {}
model_coefficients = {}

def run_regression_model(data, dep_var):
    """
    Run a standard OLS regression model with green condition and interactions,
    using the specified covariates for all models.
    """
    print(f"\nAnalyzing {dep_var}")
    
    # Prepare model data - don't include current DV if it's also a covariate
    model_data = data.dropna(subset=[dep_var]).copy()
    
    # Filter covariates to exclude the current dependent variable
    valid_covs = [cov for cov in covariates if cov != dep_var]
    
    print(f"  Observations after removing NA: {len(model_data)}")
    
    # Create formula for the regression model with interaction terms and all specified covariates
    formula = f"{dep_var} ~ IsGreen + Green_Shrub + Green_Tree"
    
    # Add all valid covariates
    if valid_covs:
        # Check which covariates are available in the data
        available_covs = [cov for cov in valid_covs if cov in model_data.columns 
                          and model_data[cov].notna().sum() > 0.8 * len(model_data)]
        if available_covs:
            formula += " + " + " + ".join(available_covs)
    
    print(f"  Using formula: {formula}")
    
    # Fit regression model
    try:
        model = smf.ols(formula, model_data)
        fit = model.fit()
        
        # Extract all fixed effects coefficients
        coefficients_df = pd.DataFrame({
            'Parameter': fit.params.index,
            'Coefficient': fit.params.values,
            'Std_Error': fit.bse.values,
            't_value': fit.tvalues.values,
            'p_value': fit.pvalues.values,
            'CI_2.5%': fit.conf_int()[0].values,
            'CI_97.5%': fit.conf_int()[1].values
        })
        
        # Add significance indicators
        coefficients_df['Significance'] = coefficients_df['p_value'].apply(
            lambda p: '***' if p < 0.001 else 
                     '**' if p < 0.01 else 
                     '*' if p < 0.05 else 
                     '.' if p < 0.1 else ''
        )
        
        # Extract main effects
        green_effect = fit.params.get('IsGreen', np.nan)
        green_se = fit.bse.get('IsGreen', np.nan)
        green_p = fit.pvalues.get('IsGreen', np.nan)
        green_ci_lower = fit.conf_int().loc['IsGreen', 0] if 'IsGreen' in fit.conf_int().index else np.nan
        green_ci_upper = fit.conf_int().loc['IsGreen', 1] if 'IsGreen' in fit.conf_int().index else np.nan
        
        # Extract shrub interaction effect
        shrub_int_effect = fit.params.get('Green_Shrub', np.nan)
        shrub_int_se = fit.bse.get('Green_Shrub', np.nan)
        shrub_int_p = fit.pvalues.get('Green_Shrub', np.nan)
        shrub_int_ci_lower = fit.conf_int().loc['Green_Shrub', 0] if 'Green_Shrub' in fit.conf_int().index else np.nan
        shrub_int_ci_upper = fit.conf_int().loc['Green_Shrub', 1] if 'Green_Shrub' in fit.conf_int().index else np.nan
        
        # Extract tree interaction effect
        tree_int_effect = fit.params.get('Green_Tree', np.nan)
        tree_int_se = fit.bse.get('Green_Tree', np.nan)
        tree_int_p = fit.pvalues.get('Green_Tree', np.nan)
        tree_int_ci_lower = fit.conf_int().loc['Green_Tree', 0] if 'Green_Tree' in fit.conf_int().index else np.nan
        tree_int_ci_upper = fit.conf_int().loc['Green_Tree', 1] if 'Green_Tree' in fit.conf_int().index else np.nan
        
        # Get covariate effects
        covariate_effects = {}
        for cov in available_covs:
            if cov in fit.params:
                covariate_effects[cov] = {
                    'coef': fit.params.get(cov, np.nan),
                    'se': fit.bse.get(cov, np.nan),
                    'p_value': fit.pvalues.get(cov, np.nan),
                    'ci_lower': fit.conf_int().loc[cov, 0] if cov in fit.conf_int().index else np.nan,
                    'ci_upper': fit.conf_int().loc[cov, 1] if cov in fit.conf_int().index else np.nan
                }
                # Add significance indicator
                p_val = covariate_effects[cov]['p_value']
                covariate_effects[cov]['significance'] = ('***' if p_val < 0.001 else 
                                                         '**' if p_val < 0.01 else 
                                                         '*' if p_val < 0.05 else 
                                                         '.' if p_val < 0.1 else '')
        
        # Calculate standardized effects
        resid_sd = np.sqrt(fit.mse_resid)  # Root MSE = residual standard deviation
        std_green_effect = green_effect / resid_sd
        std_green_ci_lower = green_ci_lower / resid_sd
        std_green_ci_upper = green_ci_upper / resid_sd
        
        std_shrub_int_effect = shrub_int_effect / resid_sd
        std_shrub_int_ci_lower = shrub_int_ci_lower / resid_sd
        std_shrub_int_ci_upper = shrub_int_ci_upper / resid_sd
        
        std_tree_int_effect = tree_int_effect / resid_sd
        std_tree_int_ci_lower = tree_int_ci_lower / resid_sd
        std_tree_int_ci_upper = tree_int_ci_upper / resid_sd
        
        # Calculate total effects for each condition type
        # In the coding scheme, IsGreen=1 for both types, and the interaction terms add the differential effects
        shrub_total_effect = green_effect + shrub_int_effect
        tree_total_effect = green_effect + tree_int_effect
        
        # Calculate standard errors for total effects using delta method approximation
        # For simplicity, we use the standard formula for the variance of a sum
        shrub_total_se = np.sqrt(green_se**2 + shrub_int_se**2 + 2*fit.cov_params().loc['IsGreen', 'Green_Shrub'])
        tree_total_se = np.sqrt(green_se**2 + tree_int_se**2 + 2*fit.cov_params().loc['IsGreen', 'Green_Tree'])
        
        # Calculate t-values and p-values for total effects
        shrub_total_t = shrub_total_effect / shrub_total_se
        tree_total_t = tree_total_effect / tree_total_se
        
        # Two-tailed p-value using t-distribution
        df_residual = fit.df_resid
        shrub_total_p = 2 * (1 - stats.t.cdf(abs(shrub_total_t), df_residual))
        tree_total_p = 2 * (1 - stats.t.cdf(abs(tree_total_t), df_residual))
        
        # Calculate confidence intervals for total effects
        t_critical = stats.t.ppf(0.975, df_residual)  # 95% CI
        shrub_total_ci_lower = shrub_total_effect - t_critical * shrub_total_se
        shrub_total_ci_upper = shrub_total_effect + t_critical * shrub_total_se
        
        tree_total_ci_lower = tree_total_effect - t_critical * tree_total_se
        tree_total_ci_upper = tree_total_effect + t_critical * tree_total_se
        
        # Calculate standardized total effects
        std_shrub_total_effect = shrub_total_effect / resid_sd
        std_shrub_total_ci_lower = shrub_total_ci_lower / resid_sd
        std_shrub_total_ci_upper = shrub_total_ci_upper / resid_sd
        
        std_tree_total_effect = tree_total_effect / resid_sd
        std_tree_total_ci_lower = tree_total_ci_lower / resid_sd
        std_tree_total_ci_upper = tree_total_ci_upper / resid_sd
        
        # Determine significance indicators
        def get_significance(p_value):
            if p_value < 0.001:
                return '***'
            elif p_value < 0.01:
                return '**'
            elif p_value < 0.05:
                return '*'
            elif p_value < 0.1:
                return '.'
            else:
                return ''
        
        green_significance = get_significance(green_p)
        shrub_int_significance = get_significance(shrub_int_p)
        tree_int_significance = get_significance(tree_int_p)
        shrub_total_significance = get_significance(shrub_total_p)
        tree_total_significance = get_significance(tree_total_p)
        
        # Calculate model fit statistics
        r2 = fit.rsquared
        adj_r2 = fit.rsquared_adj
        f_stat = fit.fvalue
        f_pvalue = fit.f_pvalue
        
        # Create model summary
        summary = {
            'dep_var': dep_var,
            'formula': formula,
            'n_obs': len(model_data),
            'n_participants': model_data['ParticipantID'].nunique(),
            # Main green effect
            'green_coef': green_effect,
            'green_se': green_se,
            'green_p': green_p,
            'green_ci_lower': green_ci_lower,
            'green_ci_upper': green_ci_upper,
            'green_std_effect': std_green_effect,
            'green_std_ci_lower': std_green_ci_lower,
            'green_std_ci_upper': std_green_ci_upper,
            'green_significance': green_significance,
            # Shrub interaction
            'shrub_int_coef': shrub_int_effect,
            'shrub_int_se': shrub_int_se,
            'shrub_int_p': shrub_int_p,
            'shrub_int_ci_lower': shrub_int_ci_lower,
            'shrub_int_ci_upper': shrub_int_ci_upper,
            'shrub_int_std_effect': std_shrub_int_effect,
            'shrub_int_std_ci_lower': std_shrub_int_ci_lower,
            'shrub_int_std_ci_upper': std_shrub_int_ci_upper,
            'shrub_int_significance': shrub_int_significance,
            # Tree interaction
            'tree_int_coef': tree_int_effect,
            'tree_int_se': tree_int_se,
            'tree_int_p': tree_int_p,
            'tree_int_ci_lower': tree_int_ci_lower,
            'tree_int_ci_upper': tree_int_ci_upper,
            'tree_int_std_effect': std_tree_int_effect,
            'tree_int_std_ci_lower': std_tree_int_ci_lower,
            'tree_int_std_ci_upper': std_tree_int_ci_upper,
            'tree_int_significance': tree_int_significance,
            # Total effects
            'shrub_total_coef': shrub_total_effect,
            'shrub_total_se': shrub_total_se,
            'shrub_total_p': shrub_total_p,
            'shrub_total_ci_lower': shrub_total_ci_lower,
            'shrub_total_ci_upper': shrub_total_ci_upper,
            'shrub_total_std_effect': std_shrub_total_effect,
            'shrub_total_std_ci_lower': std_shrub_total_ci_lower, 
            'shrub_total_std_ci_upper': std_shrub_total_ci_upper,
            'shrub_total_significance': shrub_total_significance,
            'tree_total_coef': tree_total_effect,
            'tree_total_se': tree_total_se,
            'tree_total_p': tree_total_p,
            'tree_total_ci_lower': tree_total_ci_lower,
            'tree_total_ci_upper': tree_total_ci_upper,
            'tree_total_std_effect': std_tree_total_effect,
            'tree_total_std_ci_lower': std_tree_total_ci_lower,
            'tree_total_std_ci_upper': std_tree_total_ci_upper,
            'tree_total_significance': tree_total_significance,
            # Model fit statistics
            'resid_sd': resid_sd,
            'r2': r2,
            'adj_r2': adj_r2,
            'f_stat': f_stat,
            'f_pvalue': f_pvalue,
            'df_model': fit.df_model,
            'df_resid': fit.df_resid,
            'covariate_effects': covariate_effects
        }
        
        # Print key results
        print(f"  Results:")
        print(f"    Overall Green effect: {green_effect:.3f} {green_significance} (p={green_p:.4f})")
        print(f"    Shrub interaction: {shrub_int_effect:.3f} {shrub_int_significance} (p={shrub_int_p:.4f})")
        print(f"    Tree interaction: {tree_int_effect:.3f} {tree_int_significance} (p={tree_int_p:.4f})")
        print(f"    Total Shrub effect: {shrub_total_effect:.3f} {shrub_total_significance} (p={shrub_total_p:.4f})")
        print(f"    Total Tree effect: {tree_total_effect:.3f} {tree_total_significance} (p={tree_total_p:.4f})")
        print(f"    Model fit: R²={r2:.3f}, Adj. R²={adj_r2:.3f}, F({fit.df_model:.0f},{fit.df_resid:.0f})={f_stat:.2f}, p={f_pvalue:.4f}")
        
        # Print covariate effects
        for cov, cov_effect in covariate_effects.items():
            print(f"    {cov} effect: {cov_effect['coef']:.3f} {cov_effect['significance']} (p={cov_effect['p_value']:.4f})")
        
        return fit, summary, coefficients_df
        
    except Exception as e:
        print(f"  Error fitting model: {str(e)}")
        return None, None, None

# Fit models for all dependent variables
for dep_var in dep_vars:
    model, summary, coefficients = run_regression_model(df, dep_var)
    
    if model is not None and summary is not None:
        all_models[dep_var] = model
        model_summaries[dep_var] = summary
        model_coefficients[dep_var] = coefficients

# Create Excel output with professional formatting
def create_excel_report(all_models, model_summaries, model_coefficients):
    """
    Create a professionally formatted Excel report with a summary sheet 
    and detailed individual sheets for each model.
    """
    print("\nCreating Excel report...")
    
    # Check if we have any successful models
    if not model_summaries:
        print("No successful models to report")
        return None
    
    # Create Excel file
    excel_path = 'results/green_regression_models.xlsx'
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    
    # Define styles
    header_font = Font(bold=True, size=12)
    header_fill = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")
    subheader_font = Font(bold=True, size=11)
    subheader_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    border = Border(
        left=Side(style='thin'), 
        right=Side(style='thin'), 
        top=Side(style='thin'), 
        bottom=Side(style='thin')
    )
    
    # 1. Create summary sheet for overall green effects
    green_effect_rows = []
    
    for dep_var, summary in model_summaries.items():
        # Basic row with main effects
        row = {
            'Dependent Variable': dep_var,
            'N': summary['n_obs'],
            'Participants': summary['n_participants'],
            'Overall Green Coef': summary['green_coef'],
            'SE': summary['green_se'],
            'p-value': summary['green_p'],
            'CI 2.5%': summary['green_ci_lower'],
            'CI 97.5%': summary['green_ci_upper'],
            'Std Effect (d)': summary['green_std_effect'],
            'Significance': summary['green_significance'],
            'Shrub Total Effect': summary['shrub_total_coef'],
            'Shrub p-value': summary['shrub_total_p'],
            'Shrub Sig': summary['shrub_total_significance'],
            'Tree Total Effect': summary['tree_total_coef'],
            'Tree p-value': summary['tree_total_p'],
            'Tree Sig': summary['tree_total_significance'],
            'R²': summary['r2'],
            'Adj R²': summary['adj_r2']
        }
        
        # Add all covariate effects
        for cov in covariates:
            if cov in summary['covariate_effects']:
                cov_effect = summary['covariate_effects'][cov]
                row[f'{cov} Coef'] = cov_effect['coef']
                row[f'{cov} p-value'] = cov_effect['p_value']
                row[f'{cov} Sig'] = cov_effect['significance']
            else:
                # Covariate not used in this model (might be the DV or missing)
                row[f'{cov} Coef'] = np.nan
                row[f'{cov} p-value'] = np.nan
                row[f'{cov} Sig'] = ''
        
        green_effect_rows.append(row)
    
    if green_effect_rows:
        # Sort by overall green p-value
        green_effects_df = pd.DataFrame(green_effect_rows)
        green_effects_df = green_effects_df.sort_values('p-value')
        
        # Write to Excel
        green_effects_df.to_excel(writer, sheet_name='Green Effects', index=False)
        
        # Format the Green Effects sheet
        workbook = writer.book
        worksheet = writer.sheets['Green Effects']
        
        # Apply formatting to header row
        for col in range(1, len(green_effects_df.columns) + 1):
            cell = worksheet.cell(row=1, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = border
        
        # Set column widths
        for col in range(1, len(green_effects_df.columns) + 1):
            col_letter = get_column_letter(col)
            worksheet.column_dimensions[col_letter].width = 15
        
        # Add auto-filter
        worksheet.auto_filter.ref = worksheet.dimensions
    
    # 2. Create summary sheet for interaction effects
    interaction_rows = []
    
    for dep_var, summary in model_summaries.items():
        # Shrub interaction row
        shrub_row = {
            'Dependent Variable': dep_var,
            'Interaction Type': 'Shrub',
            'Interaction Coef': summary['shrub_int_coef'],
            'SE': summary['shrub_int_se'],
            'p-value': summary['shrub_int_p'],
            'CI 2.5%': summary['shrub_int_ci_lower'],
            'CI 97.5%': summary['shrub_int_ci_upper'],
            'Std Effect (d)': summary['shrub_int_std_effect'],
            'Significance': summary['shrub_int_significance'],
            'Total Effect': summary['shrub_total_coef'],
            'Total p-value': summary['shrub_total_p'],
            'Total Sig': summary['shrub_total_significance']
        }
        interaction_rows.append(shrub_row)
        
        # Tree interaction row
        tree_row = {
            'Dependent Variable': dep_var,
            'Interaction Type': 'Tree',
            'Interaction Coef': summary['tree_int_coef'],
            'SE': summary['tree_int_se'],
            'p-value': summary['tree_int_p'],
            'CI 2.5%': summary['tree_int_ci_lower'],
            'CI 97.5%': summary['tree_int_ci_upper'],
            'Std Effect (d)': summary['tree_int_std_effect'],
            'Significance': summary['tree_int_significance'],
            'Total Effect': summary['tree_total_coef'],
            'Total p-value': summary['tree_total_p'],
            'Total Sig': summary['tree_total_significance']
        }
        interaction_rows.append(tree_row)
    
    if interaction_rows:
        # Create DataFrame and sort
        interactions_df = pd.DataFrame(interaction_rows)
        interactions_df = interactions_df.sort_values(['Dependent Variable', 'p-value'])
        
        # Write to Excel
        interactions_df.to_excel(writer, sheet_name='Interaction Effects', index=False)
        
        # Format the Interaction Effects sheet
        worksheet = writer.sheets['Interaction Effects']
        
        # Apply formatting to header row
        for col in range(1, len(interactions_df.columns) + 1):
            cell = worksheet.cell(row=1, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = border
        
        # Set column widths
        for col in range(1, len(interactions_df.columns) + 1):
            col_letter = get_column_letter(col)
            worksheet.column_dimensions[col_letter].width = 15
        
        # Add auto-filter
        worksheet.auto_filter.ref = worksheet.dimensions
    
    # 3. Create summary sheet for covariate effects
    covariate_effect_rows = []
    
    for dep_var, summary in model_summaries.items():
        for cov, cov_effect in summary.get('covariate_effects', {}).items():
            row = {
                'Dependent Variable': dep_var,
                'Covariate': cov,
                'Coefficient': cov_effect.get('coef', np.nan),
                'Std Error': cov_effect.get('se', np.nan),
                'p-value': cov_effect.get('p_value', np.nan),
                'CI 2.5%': cov_effect.get('ci_lower', np.nan),
                'CI 97.5%': cov_effect.get('ci_upper', np.nan),
                'Significance': cov_effect.get('significance', '')
            }
            covariate_effect_rows.append(row)
    
    if covariate_effect_rows:
        # Create DataFrame and sort
        covariate_effects_df = pd.DataFrame(covariate_effect_rows)
        covariate_effects_df = covariate_effects_df.sort_values(['Covariate', 'p-value'])
        
        # Write to Excel
        covariate_effects_df.to_excel(writer, sheet_name='Covariate Effects', index=False)
        
        # Format the Covariate Effects sheet
        worksheet = writer.sheets['Covariate Effects']
        
        # Apply formatting to header row
        for col in range(1, len(covariate_effects_df.columns) + 1):
            cell = worksheet.cell(row=1, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = border
        
        # Set column widths
        for col in range(1, len(covariate_effects_df.columns) + 1):
            col_letter = get_column_letter(col)
            worksheet.column_dimensions[col_letter].width = 15
        
        # Add auto-filter
        worksheet.auto_filter.ref = worksheet.dimensions
    
    # 4. Individual sheets for each model with detailed statistics
    for dep_var in model_summaries.keys():
        # Create safe sheet name (max 31 chars for Excel)
        safe_name = dep_var
        if len(safe_name) > 30:
            safe_name = safe_name[:27] + "..."
        
        # Get model details
        summary = model_summaries[dep_var]
        coefficients = model_coefficients.get(dep_var, None)
        
        # Create data for detailed model sheet
        detailed_data = []
        
        # Model summary section
        detailed_data.append(["Model Summary", "", "", "", "", ""])
        detailed_data.append(["Dependent Variable", dep_var, "", "", "", ""])
        detailed_data.append(["Formula", summary['formula'], "", "", "", ""])
        detailed_data.append(["Observations", summary['n_obs'], "", "", "", ""])
        detailed_data.append(["Participants", summary['n_participants'], "", "", "", ""])
        detailed_data.append(["R²", summary['r2'], "", "", "", ""])
        detailed_data.append(["Adjusted R²", summary['adj_r2'], "", "", "", ""])
        detailed_data.append(["F-statistic", f"F({summary['df_model']:.0f},{summary['df_resid']:.0f})={summary['f_stat']:.2f}, p={summary['f_pvalue']:.4f}", "", "", "", ""])
        detailed_data.append(["Residual Std Dev", summary['resid_sd'], "", "", "", ""])
        detailed_data.append(["", "", "", "", "", ""])  # Empty row
        
        # Overall green effect
        detailed_data.append(["Overall Green Effect", "", "", "", "", ""])
        detailed_data.append(["Coefficient", "Std Error", "p-value", "CI 2.5%", "CI 97.5%", "Significance"])
        detailed_data.append([
            summary['green_coef'],
            summary['green_se'],
            summary['green_p'],
            summary['green_ci_lower'],
            summary['green_ci_upper'],
            summary['green_significance']
        ])
        detailed_data.append(["", "", "", "", "", ""])  # Empty row
        
        # Interaction effects section
        detailed_data.append(["Interaction Effects", "", "", "", "", ""])
        detailed_data.append(["Type", "Coefficient", "Std Error", "p-value", "CI 2.5%", "CI 97.5%"])
        
        # Shrub interaction
        detailed_data.append([
            "Shrub",
            summary['shrub_int_coef'],
            summary['shrub_int_se'],
            summary['shrub_int_p'],
            summary['shrub_int_ci_lower'],
            summary['shrub_int_ci_upper']
        ])
        
        # Tree interaction
        detailed_data.append([
            "Tree",
            summary['tree_int_coef'],
            summary['tree_int_se'],
            summary['tree_int_p'],
            summary['tree_int_ci_lower'],
            summary['tree_int_ci_upper']
        ])
        detailed_data.append(["", "", "", "", "", ""])  # Empty row
        
        # Total effects section
        detailed_data.append(["Total Effects by Condition", "", "", "", "", ""])
        detailed_data.append(["Condition", "Coefficient", "Std Error", "p-value", "CI 2.5%", "CI 97.5%"])
        
        # Shrub total effect
        detailed_data.append([
            "Shrub",
            summary['shrub_total_coef'],
            summary['shrub_total_se'],
            summary['shrub_total_p'],
            summary['shrub_total_ci_lower'],
            summary['shrub_total_ci_upper']
        ])
        
        # Tree total effect
        detailed_data.append([
            "Tree",
            summary['tree_total_coef'],
            summary['tree_total_se'],
            summary['tree_total_p'],
            summary['tree_total_ci_lower'],
            summary['tree_total_ci_upper']
        ])
        detailed_data.append(["", "", "", "", "", ""])  # Empty row
        
        # Standardized effect section
        detailed_data.append(["Standardized Effects (Cohen's d)", "", "", "", "", ""])
        detailed_data.append(["Effect Type", "Std Effect", "CI 2.5%", "CI 97.5%", "", ""])
        
        # Overall green
        detailed_data.append([
            "Overall Green",
            summary['green_std_effect'],
            summary['green_std_ci_lower'],
            summary['green_std_ci_upper'],
            "", ""
        ])
        
        # Shrub interaction
        detailed_data.append([
            "Shrub Interaction",
            summary['shrub_int_std_effect'],
            summary['shrub_int_std_ci_lower'],
            summary['shrub_int_std_ci_upper'],
            "", ""
        ])
        
        # Tree interaction
        detailed_data.append([
            "Tree Interaction",
            summary['tree_int_std_effect'],
            summary['tree_int_std_ci_lower'],
            summary['tree_int_std_ci_upper'],
            "", ""
        ])
        
        # Total effects
        detailed_data.append([
            "Shrub Total",
            summary['shrub_total_std_effect'],
            summary['shrub_total_std_ci_lower'],
            summary['shrub_total_std_ci_upper'],
            "", ""
        ])
        
        detailed_data.append([
            "Tree Total",
            summary['tree_total_std_effect'],
            summary['tree_total_std_ci_lower'],
            summary['tree_total_std_ci_upper'],
            "", ""
        ])
        detailed_data.append(["", "", "", "", "", ""])  # Empty row
        
        # Covariate effects section
        detailed_data.append(["Covariate Effects", "", "", "", "", ""])
        detailed_data.append(["Covariate", "Coefficient", "Std Error", "p-value", "Significance", "CI 95%"])
        
        for cov, cov_effect in summary.get('covariate_effects', {}).items():
            ci_str = f"[{cov_effect.get('ci_lower', np.nan):.3f}, {cov_effect.get('ci_upper', np.nan):.3f}]"
            detailed_data.append([
                cov,
                cov_effect.get('coef', np.nan),
                cov_effect.get('se', np.nan),
                cov_effect.get('p_value', np.nan),
                cov_effect.get('significance', ''),
                ci_str
            ])
        
        # Create DataFrame and write to Excel
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_excel(writer, sheet_name=safe_name, index=False, header=False)
        
        # Format the detailed sheet
        worksheet = writer.sheets[safe_name]
        
        # Apply formatting to section headers and data
        for row_idx, row_data in enumerate(detailed_data, 1):
            if row_data[0] in ["Model Summary", "Overall Green Effect", "Interaction Effects", 
                              "Total Effects by Condition", "Standardized Effects (Cohen's d)", 
                              "Covariate Effects"]:
                # Section header
                cell = worksheet.cell(row=row_idx, column=1)
                cell.font = header_font
                cell.fill = subheader_fill
                
                # Format entire row
                for col_idx in range(1, 7):
                    cell = worksheet.cell(row=row_idx, column=col_idx)
                    cell.font = header_font
                    cell.fill = subheader_fill
            
            elif row_data[0] in ["Type", "Condition", "Effect Type", "Covariate"] or (row_idx > 1 and detailed_data[row_idx-2][0] == "Overall Green Effect" and row_data[0] == "Coefficient"):
                # Column header for data sections
                for col_idx in range(1, 7):
                    if col_idx <= len(row_data) and row_data[col_idx-1]:
                        cell = worksheet.cell(row=row_idx, column=col_idx)
                        cell.font = subheader_font
        
        # Set column widths
        for col in range(1, 7):
            col_letter = get_column_letter(col)
            worksheet.column_dimensions[col_letter].width = 18
    
    # 5. Add a note about the model
    note_df = pd.DataFrame({
        'Note': [
            'MODEL INTERPRETATION:',
            '',
            'This model uses a standard regression approach to analyze the effect of green environments on various outcomes.',
            'For each dependent variable, the model estimates:',
            '',
            '1. Overall Green Effect: The general effect of any green environment vs control.',
            '',
            '2. Interaction Effects: How different types of green environments (shrubs vs trees) differ from the average green effect.',
            '   - Green_Shrub: Differential effect of shrubs compared to the average green effect',
            '   - Green_Tree: Differential effect of trees compared to the average green effect',
            '',
            '3. Total Effects: The complete effect of each specific environment type.',
            '   - Shrub Total = Overall Green + Shrub Interaction',
            '   - Tree Total = Overall Green + Tree Interaction',
            '',
            '4. Standardized Effects (Cohen\'s d): Effect sizes standardized by the residual standard deviation.',
            '   - Values around 0.2 are considered small',
            '   - Values around 0.5 are considered medium',
            '   - Values around 0.8 are considered large',
            '',
            'COVARIATES:',
            'RunOrder_num: Controls for potential order effects',
            'Age: Controls for age differences between participants',
            'Gender_num: Controls for gender differences',
            'MRExperience_num: Controls for prior experience with mixed reality',
            'VO2max: Controls for fitness level',
            'AvgSkinTemp: Controls for skin temperature',
            'Presence_Avg: Controls for sense of presence',
            '',
            'Significance codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1'
        ]
    })
    note_df.to_excel(writer, sheet_name='Notes', index=False)
    
    # Close the Excel writer
    writer.close()
    
    print(f"Excel report saved to {excel_path}")
    return excel_path

# Generate Excel report with detailed individual model sheets
excel_file = create_excel_report(all_models, model_summaries, model_coefficients)

print("\nAnalysis complete!")