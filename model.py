import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.utils import get_column_letter
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Create directory for results
os.makedirs('results', exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv('data/Combined_Joined_Data.csv')

# Print columns for reference
print("\nAvailable columns:")
columns_list = df.columns.tolist()
print(columns_list)

# Define column mappings with correct names from the data
column_mappings = {
    # Basic info columns
    'experiment ID (participant number_condition)': 'ExperimentID',
    'participant number': 'ParticipantID',
    'condition': 'Condition_num',
    'run order': 'RunOrder_num',
    'Past exepriences with XR': 'MRExperience_num',
    'Gender': 'Gender_num',
    'Age': 'Age',
    'VO2max': 'VO2max',
    'avg_close_to_skin_temp': 'AvgSkinTemp',
    'min_heart_rate': 'MinHR',
    'Average heart rate (bpm)': 'AvgHR',
    'Recalculated_Speed_km_per_h': 'Speed_kmh',
    'Duration_seconds': 'Duration_s',
    
    # Key measure columns with correct names from the data
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
    'Sense of belonging': 'Belonging'
}

# Rename Columns
rename_dict = {k: v for k, v in column_mappings.items() if k in df.columns}
df = df.rename(columns=rename_dict)
# Print renamed columns for reference
print("\nRenamed columns:")
print(df.columns.tolist())

# Convert numeric columns to float
numeric_columns = [col for col in df.columns if col in set(column_mappings.values())]
print(f"\nProcessing {len(numeric_columns)} numeric columns.")
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
# Ensure Condition_num is numeric
if 'Condition_num' in df.columns:
    df['Condition_num'] = pd.to_numeric(df['Condition_num'], errors='coerce')
else:
    print("ERROR: Condition_num column not found!")
    exit(1)


# Determine if a participant was in tree group (1) or shrub group (0)
participant_groups = {}

# For each participant, check their condition assignments
for participant_id in df['ParticipantID'].unique():
    participant_data = df[df['ParticipantID'] == participant_id]
    
    # Check if they have tree runs (condition 2)
    has_tree_runs = (participant_data['Condition_num'] == 2).any()
    
    # Check if they have shrub runs (condition 1)
    has_shrub_runs = (participant_data['Condition_num'] == 1).any()
    
    # Assign group (1 for tree group, 0 for shrub group)
    if has_tree_runs and not has_shrub_runs:
        participant_groups[participant_id] = 1  # Tree group
    elif has_shrub_runs and not has_tree_runs:
        participant_groups[participant_id] = 0  # Shrub group

# Create RunGroupType column (Tree=1, Shrub=0)
df['RunGroupType'] = df['ParticipantID'].map(participant_groups)

# Create green condition indicator (1=green, 0=control)
df['IsGreen'] = (df['Condition_num'] > 0).astype(int)

# Create interaction term
df['IsGreen_RunGroupType'] = df['IsGreen'] * df['RunGroupType']

# Print condition counts
print("\nCondition counts:")
print(f"Condition_num:\n{df['Condition_num'].value_counts().sort_index()}")
print(f"\nIsGreen (1=green, 0=control):\n{df['IsGreen'].value_counts()}")
print(f"\nRunGroupType (1=tree group, 0=shrub group):\n{df['RunGroupType'].value_counts()}")

# Calculate composite variables
print("\nCalculating composite variables...")
# Presence composite
presence_items = ['Immersed', 'PhysicallyPresent', 'UseObjects', 'AbilityToDo']
df['Presence_Avg'] = df[presence_items].mean(axis=1)
# Perceived Restorativeness Scale
prs_items = ['Relaxation', 'Curiosity', 'OrderInSpace', 'EasyNavigation', 'ComfortableEnv']
df['PRS_Avg'] = df[prs_items].mean(axis=1)
# Place Attachment
pa_items = ['Identification', 'Belonging']
df['PlaceAttach_Avg'] = df[pa_items].mean(axis=1)
# Positive Affect
positive_affect_items = ['Enjoyment', 'Satisfaction', 'Healthy', 'LightWeighted']
df['PositiveAffect_Avg'] = df[positive_affect_items].mean(axis=1)
# Negative Affect
negative_affect_items = ['Fatigue', 'Tense', 'Anxious', 'Angry', 'Irritated', 'Sluggish']
df['NegativeAffect_Avg'] = df[negative_affect_items].mean(axis=1)

# Define all dependent variables
dep_vars = [
    'PRS_Avg', 'PlaceAttach_Avg', 'PositiveAffect_Avg', 'NegativeAffect_Avg', 
    'Duration_s', 'RPE', 'Enjoyment', 'Satisfaction', 'Relaxation', 'Healthy', 'LightWeighted', 
    'Fatigue', 'Tense', 'Anxious', 'Angry', 'Irritated', 'Sluggish', 'Concentration', 'MoveFreely', 'Exhaustion', 
    'Presence', 'Immersed', 'PhysicallyPresent', 'UseObjects', 'AbilityToDo',
    'Challenging', 'PerceivedGreenness', 'MinHR', 'Speed_kmh', 'AvgHR'
]
print(f"\nAnalyzing {len(dep_vars)} dependent variables: {dep_vars}")

# Define covariates
covariates = ['RunOrder_num', 'Age', 'Gender_num', 'VO2max', 'AvgSkinTemp', 'Presence_Avg']
print(f"Using {len(covariates)} covariates: {covariates}")

# Storage for results
all_results = {}

def get_significance_symbol(p_value):
    """Return significance symbols based on p-value"""
    if pd.isna(p_value):
        return ''
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    elif p_value < 0.1:
        return '†'
    else:
        return ''

def run_mixed_model(data, dep_var):
    """Run a linear mixed effects model"""
    print(f"\nAnalyzing {dep_var}")
    
    # Prepare model data
    model_data = data.dropna(subset=[dep_var]).copy()
    
    # Filter covariates to exclude current DV
    valid_covs = [cov for cov in covariates if cov != dep_var]
    
    # Skip model if insufficient data
    if len(model_data) < 10:
        print(f"  Skipping: Not enough data (N={len(model_data)})")
        return None
    
    # Ensure ParticipantID is a string for grouping
    model_data['ParticipantID'] = model_data['ParticipantID'].astype(str)
    
    # Create formula - factorial design with covariates
    formula = f"{dep_var} ~ IsGreen + RunGroupType + IsGreen:RunGroupType"
    
    # Add valid covariates if they have enough non-missing values
    available_covs = []
    if valid_covs:
        available_covs = [cov for cov in valid_covs if cov in model_data.columns 
                         and model_data[cov].notna().sum() > 0.8 * len(model_data)]
        if available_covs:
            formula += " + " + " + ".join(available_covs)
    
    print(f"  Formula: {formula}")
    print(f"  N = {len(model_data)}, Participants = {model_data['ParticipantID'].nunique()}")
    
    try:
        # Run mixed model with random intercept for participant
        model = smf.mixedlm(formula, model_data, groups=model_data["ParticipantID"])
        fit = model.fit(reml=True)
        
        # Extract key parameters
        # IsGreen effect (effect in shrub group, the reference level)
        green_effect = fit.params.get('IsGreen', np.nan)
        green_p = fit.pvalues.get('IsGreen', np.nan)
        green_se = fit.bse.get('IsGreen', np.nan)
        
        # RunGroupType effect (difference between tree and shrub groups)
        group_effect = fit.params.get('RunGroupType', np.nan)
        group_p = fit.pvalues.get('RunGroupType', np.nan)
        group_se = fit.bse.get('RunGroupType', np.nan)
        
        # Interaction effect (additional effect of IsGreen in the tree group)
        interaction = fit.params.get('IsGreen:RunGroupType', np.nan)
        interaction_p = fit.pvalues.get('IsGreen:RunGroupType', np.nan)
        interaction_se = fit.bse.get('IsGreen:RunGroupType', np.nan)
        
        # Get random effects variance
        random_var = float(fit.cov_re.iloc[0, 0])
        residual_var = fit.scale
        icc = random_var / (random_var + residual_var)
        
        # Calculate R-squared values for mixed models (simpler approach)
        # Using the method from the 'performance' package in R
        
        # Calculate simplified R² values
        total_var = model_data[dep_var].var()
        residuals = model_data[dep_var] - fit.fittedvalues
        residual_var_actual = np.var(residuals)
        
        # Marginal R² - fixed effects only
        r2_marginal = 1 - (residual_var_actual / total_var)
        
        # Conditional R² - fixed + random effects 
        # Note: This is an approximation
        r2_conditional = 1 - (residual_var / total_var)
        
        # Make sure R² values are within bounds
        r2_marginal = max(0, min(1, r2_marginal))
        r2_conditional = max(0, min(1, r2_conditional))
        
        # Calculate AIC and BIC manually
        # Count parameters: fixed effects + 1 for random intercept variance + 1 for residual variance
        n_fixed_params = len(fit.params)
        k = n_fixed_params + 2  # +2 for random intercept variance and residual variance
        n = len(model_data)
        loglik = fit.llf
        aic = -2 * loglik + 2 * k
        bic = -2 * loglik + k * np.log(n)
        
        # Get covariate effects
        covariate_results = {}
        for cov in available_covs:
            if cov in fit.params:
                covariate_results[cov] = {
                    'coef': fit.params.get(cov, np.nan),
                    'p': fit.pvalues.get(cov, np.nan),
                    'sig': get_significance_symbol(fit.pvalues.get(cov, np.nan))
                }
        
        # Create results dictionary
        results = {
            'n': len(model_data),
            'participants': model_data['ParticipantID'].nunique(),
            'formula': formula,
            'green_effect': green_effect,
            'green_p': green_p,
            'green_sig': get_significance_symbol(green_p),
            'group_effect': group_effect,
            'group_p': group_p,
            'group_sig': get_significance_symbol(group_p),
            'interaction': interaction,
            'interaction_p': interaction_p,
            'interaction_sig': get_significance_symbol(interaction_p),
            'participant_icc': icc,
            'r2_marginal': r2_marginal,
            'r2_conditional': r2_conditional,
            'aic': aic,
            'bic': bic,
            'loglik': loglik,
            'covariates': covariate_results,
            'fit': fit  # Store the full model fit for reference
        }
        
        # Print key results
        print(f"  Results:")
        print(f"    IsGreen effect (in shrub group): {green_effect:.3f} {get_significance_symbol(green_p)} (p={green_p:.4f})")
        print(f"    RunGroupType effect: {group_effect:.3f} {get_significance_symbol(group_p)} (p={group_p:.4f})")
        print(f"    Tree_Interaction effect: {interaction:.3f} {get_significance_symbol(interaction_p)} (p={interaction_p:.4f})")
        print(f"    Participant_ICC: {icc:.3f}, R² Marginal: {r2_marginal:.3f}, R² Conditional: {r2_conditional:.3f}")
        print(f"    AIC: {aic:.2f}, BIC: {bic:.2f}")
        
        return results
    except Exception as e:
        print(f"  Error: {str(e)}")
        return None

def create_excel_report(results):
    """Create a clean, simple Excel report with results"""
    print("\nCreating Excel report...")
    
    if not results:
        print("No results to report")
        return
    
    excel_path = 'results/green_environment_models.xlsx'
    writer = pd.ExcelWriter(excel_path, engine='openpyxl')
    
    # Create styles
    header_font = Font(bold=True, size=11)
    header_fill = PatternFill(start_color="E0EBF5", end_color="E0EBF5", fill_type="solid")
    border = Border(
        left=Side(style='thin'), 
        right=Side(style='thin'), 
        top=Side(style='thin'), 
        bottom=Side(style='thin')
    )
    
    # Get all possible covariates across all models
    all_covariates = set()
    for res in results.values():
        for cov in res.get('covariates', {}):
            all_covariates.add(cov)
    all_covariates = sorted(list(all_covariates))
    
    # Create summary rows with all effects and p-values
    summary_rows = []
    for dv, res in results.items():
        # Start with basic info
        row = {
            'Dependent Variable': dv,
            'N': res['n'],
            'Participants': res['participants'],
            # Main effects with values and p-values separately
            'IsGreen': res['green_effect'],
            'IsGreen p': res['green_p'],
            'IsGreen sig': res['green_sig'],
            'RunGroupType': res['group_effect'],
            'RunGroupType p': res['group_p'], 
            'RunGroupType sig': res['group_sig'],
            'Tree_Interaction': res['interaction'],
            'Tree_Interaction p': res['interaction_p'],
            'Tree_Interaction sig': res['interaction_sig'],
        }
        
        # Add all possible covariates (with empty values for those not in this model)
        for cov in all_covariates:
            if cov in res.get('covariates', {}):
                cov_effect = res['covariates'][cov]
                row[f'{cov}'] = cov_effect['coef']
                row[f'{cov} p'] = cov_effect['p']
                row[f'{cov} sig'] = cov_effect['sig']
            else:
                row[f'{cov}'] = np.nan
                row[f'{cov} p'] = np.nan
                row[f'{cov} sig'] = ''
        
        # Add model fit metrics at the end
        row['R² Marginal'] = res['r2_marginal']
        row['R² Conditional'] = res['r2_conditional']
        row['Participant_ICC'] = res['participant_icc']
        row['AIC'] = res['aic']
        row['BIC'] = res['bic']
        
        summary_rows.append(row)
    
    # Sort by statistical significance of IsGreen effect
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(['IsGreen sig', 'IsGreen p'], 
                                       key=lambda x: pd.Categorical(summary_df['IsGreen sig'], 
                                                                   categories=['***', '**', '*', '.', ''], 
                                                                   ordered=True),
                                       ascending=[False, True])
    
    # Write summary sheet
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # Format summary sheet
    ws = writer.sheets['Summary']
    
    # Create grouped headers
    ws.insert_rows(1)
    
    # Calculate column positions
    covariate_start_col = 13
    covariate_end_col = covariate_start_col + (len(all_covariates) * 3) - 1
    model_fit_start_col = covariate_end_col + 1
    
    # Set main header cells
    main_header_cells = {
        'A1': 'Basic Info',
        'D1': 'IsGreen Effect',
        'G1': 'RunGroupType Effect',
        'J1': 'Tree_Interaction Effect'
    }
    
    # Add covariate headers
    col_index = covariate_start_col
    for cov in all_covariates:
        col_letter = get_column_letter(col_index)
        main_header_cells[f'{col_letter}1'] = f'{cov} Effect'
        col_index += 3
    
    # Add model fit header
    model_fit_col = get_column_letter(model_fit_start_col)
    main_header_cells[f'{model_fit_col}1'] = 'Model Fit'
    
    # Apply main headers
    for cell_ref, value in main_header_cells.items():
        ws[cell_ref] = value
        ws[cell_ref].font = header_font
        ws[cell_ref].fill = header_fill
        ws[cell_ref].alignment = Alignment(horizontal='center')
    
    # Merge header cells
    header_merges = {
        'A1:C1': 'Basic Info',
        'D1:F1': 'IsGreen Effect',
        'G1:I1': 'RunGroupType Effect',
        'J1:L1': 'Tree_Interaction Effect'
    }
    
    # Add covariate merge ranges
    col_index = covariate_start_col
    for cov in all_covariates:
        start_letter = get_column_letter(col_index)
        end_letter = get_column_letter(col_index + 2)
        header_merges[f'{start_letter}1:{end_letter}1'] = f'{cov} Effect'
        col_index += 3
    
    # Add model fit merge range
    model_fit_start_letter = get_column_letter(model_fit_start_col)
    model_fit_end_letter = get_column_letter(model_fit_start_col + 4)  # Now 5 columns: R² Marginal, R² Conditional, ICC, AIC, BIC
    header_merges[f'{model_fit_start_letter}1:{model_fit_end_letter}1'] = 'Model Fit'
    
    # Apply merges
    for merge_range, _ in header_merges.items():
        ws.merge_cells(merge_range)
    
    # Format all cells in the header rows
    for col in range(1, len(summary_df.columns) + 1):
        # Format second row (original header)
        cell = ws.cell(row=2, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = border
        
        # Set column width
        ws.column_dimensions[get_column_letter(col)].width = 12
    
    # Special width for first column
    ws.column_dimensions['A'].width = 20
    
    # Add filter
    ws.auto_filter.ref = f"A2:{get_column_letter(len(summary_df.columns))}2"
    
    # Individual model sheets (compact)
    for dv, res in results.items():
        # Create safe sheet name (max 31 chars)
        sheet_name = dv[:30]
        
        # Create compact data for model sheet
        model_data = [
            ["Mixed Effects Model", dv, "", "", ""],
            ["N:", f"{res['n']}", "Participants:", f"{res['participants']}", ""],
            ["", "", "", "", ""],
            ["Effect", "Estimate", "SE", "p-value", "Significance"],
            ["IsGreen", f"{res['green_effect']:.4f}", f"{res['fit'].bse.get('IsGreen', np.nan):.4f}", f"{res['green_p']:.4f}", res['green_sig']],
            ["RunGroupType", f"{res['group_effect']:.4f}", f"{res['fit'].bse.get('RunGroupType', np.nan):.4f}", f"{res['group_p']:.4f}", res['group_sig']],
            ["Tree_Interaction", f"{res['interaction']:.4f}", f"{res['fit'].bse.get('IsGreen:RunGroupType', np.nan):.4f}", f"{res['interaction_p']:.4f}", res['interaction_sig']],
            ["", "", "", "", ""],
            ["Model Fit", "", "", "", ""],
            ["R² Marginal:", f"{res['r2_marginal']:.4f}", "R² Conditional:", f"{res['r2_conditional']:.4f}", ""],
            ["Participant_ICC:", f"{res['participant_icc']:.4f}", "Log-Likelihood:", f"{res['loglik']:.2f}", ""],
            ["AIC:", f"{res['aic']:.2f}", "BIC:", f"{res['bic']:.2f}", ""]
        ]
        
        # Add covariate section if there are any covariates
        if res.get('covariates'):
            model_data.append(["", "", "", "", ""])
            model_data.append(["Covariates", "Estimate", "SE", "p-value", "Significance"])
            
            for cov, cov_res in res.get('covariates', {}).items():
                model_data.append([
                    cov, 
                    f"{cov_res['coef']:.4f}", 
                    f"{res['fit'].bse.get(cov, np.nan):.4f}",
                    f"{cov_res['p']:.4f}", 
                    cov_res['sig']
                ])
        
        # Convert to DataFrame and write to sheet
        model_df = pd.DataFrame(model_data)
        model_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        
        # Format model sheet
        ws = writer.sheets[sheet_name]
        
        # Format headers and section titles
        for row_idx, row_data in enumerate(model_data, 1):
            if row_idx == 1:  # Title row
                cell = ws.cell(row=row_idx, column=1)
                cell.font = Font(bold=True, size=12)
                
                cell = ws.cell(row=row_idx, column=2)
                cell.font = Font(bold=True, size=12)
                
            elif row_data[0] in ["Effect", "Model Fit", "Covariates"]:  # Section headers
                cell = ws.cell(row=row_idx, column=1)
                cell.font = header_font
                cell.fill = header_fill
                
                # Format header row
                for col_idx in range(1, 6):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    cell.font = header_font
                    cell.fill = header_fill
            
        # Set column widths
        ws.column_dimensions['A'].width = 15
        ws.column_dimensions['B'].width = 12
        ws.column_dimensions['C'].width = 15
        ws.column_dimensions['D'].width = 15
        ws.column_dimensions['E'].width = 12
    
    # Save Excel file
    writer.close()
    print(f"Excel report saved to {excel_path}")

# Run models for all dependent variables
print("\nRunning mixed effects models for all dependent variables...")
for dv in dep_vars:
    results = run_mixed_model(df, dv)
    if results:
        all_results[dv] = results

# Generate the report
create_excel_report(all_results)

print("\nAnalysis complete!")