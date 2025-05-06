import pandas as pd

df = pd.read_csv('data/Combined_Joined_Data.csv')

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

print(df.head())

print(df.info())

print(df.describe())

print(df.columns)


'''ndex(['experiment ID (participant number_condition)',
       'Past exepriences with XR',
       'What is the research objective in your eyes', 'Timestamp',
       'participant number', 'condition', 'run order', 'start time',
       'האם הנך מסכים.ה להשתתף במחקר?', 'RPE', 'Satisfaction', 'Enjoyment',
       'Challenging', 'Concentration', 'Feeling of presence',
       'Ability to move freely', 'Fatigue', 'Exhaustion', 'Sluggish',
       'Feeling light weighted',
       'Relaxation and losing oneself in the environment',
       'Curiosity and fascination', 'Order in space', 'Easy navigation',
       'Comfortable environment', 'Tense', 'Anxious', 'Healthy', 'Angry',
       'Irritated', 'Immersed', 'Physically present',
       'Movement around and use of objects', 'Ability to do things',
       'Identification', 'Greenes of the environment', 'Sense of belonging',
       'Interesting things to observe', 'Monotonous', 'Not well-maintained',
       'Gender', 'Age', 'Name', 'Sport', 'Date', 'Start time Watch',
       'Duration', 'Total distance (km)', 'Average heart rate (bpm)',
       'Average speed (km/h)', 'Max speed (km/h)', 'Average pace (min/km)',
       'Max pace (min/km)', 'Calories', 'Fat percentage of calories(%)',
       'Carbohydrate percentage of calories(%)',
       'Protein percentage of calories(%)', 'Average cadence (rpm)',
       'Average stride length (cm)', 'Running index', 'Training load',
       'Ascent (m)', 'Descent (m)', 'Average power (W)', 'Max power (W)',  'VO2max', '...30',
       'Duration_seconds', 'Recalculated_Pace_min_per_km',
       'Recalculated_Speed_km_per_h', 'Run', 'min_heart_rate',
       'avg_close_to_skin_temp'],
      dtype='object')
'''

independent_variables = ['Condition_num']  # control: 0, green: 1 or 2, difference between 0-1, and between 0-2

# Update covariates to match R code exactly
covariates = ['RunOrder_num', 'Age', 'Gender_num', 'MRExperience_num', 'VO2max', 'AvgSkinTemp', 'TimeOfDay', 'Presence_Avg']

# COVARIATES
# takes an average of the items (1-5)
presence_items = ['Immersed', 'PhysicallyPresent', 'UseObjects', 'AbilityToDo']


# DEPENDENT VARIABLES
# Calculate row-wise averages (per participant) for each scale
prs_items = ['Relaxation', 'Curiosity', 'OrderInSpace', 'EasyNavigation', 'ComfortableEnv']
df['PRS_Avg'] = df[prs_items].mean(axis=1)  # axis=1 means row-wise average

pa_items = ['Identification', 'Belonging']
df['PlaceAttach_Avg'] = df[pa_items].mean(axis=1)  # row-wise average

positive_affect_items = ['Enjoyment', 'Satisfaction', 'Healthy', 'LightWeighted']
df['PositiveAffect_Avg'] = df[positive_affect_items].mean(axis=1)  # row-wise average

negative_affect_items = ['Fatigue', 'Tense', 'Anxious', 'Angry', 'Irritated', 'Sluggish']
df['NegativeAffect_Avg'] = df[negative_affect_items].mean(axis=1)  # row-wise average

print('PRS_Avg', df['PRS_Avg'].mean())
print('PlaceAttach_Avg', df['PlaceAttach_Avg'].mean())
print('PositiveAffect_Avg', df['PositiveAffect_Avg'].mean())
print('NegativeAffect_Avg', df['NegativeAffect_Avg'].mean())

# use as is for dependent variables
duration = 'Duration_s'

