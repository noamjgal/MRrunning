# --- 1. Load Necessary Libraries ---

# Check if packages are installed, install if not, then load
packages <- c("lme4", "lmerTest", "emmeans", "dplyr", "tidyr", "ggplot2",
              "readr", "performance", "see", "lubridate", "pbkrtest", "psych", "car") # Added car for fallback ANOVA

print("Loading required packages...")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    print(paste("Installing package:", pkg))
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}
print("Packages loaded.")

# --- START: Redirect Console Output to File ---
output_filename <- "full_analysis_output_v3.txt" # Changed filename for new version
print(paste("Redirecting console output to:", output_filename))
# sink(output_filename, split = TRUE) # Use split=TRUE to see output on console AND file
sink(output_filename, split = FALSE) # Original: Only output to file
print(paste("Analysis started at:", Sys.time()))
# --- END: Redirect Console Output to File ---


# --- 2. Load and Prepare Data ---

# Explicitly load the combined data from the CSV file
combined_data_file <- "Combined_Joined_Data.csv"
if (!file.exists(combined_data_file)) {
  print(paste("Error: Data file not found:", combined_data_file))
  sink() # Stop sinking if file not found
  stop(paste("Data file not found:", combined_data_file, "- Please ensure it's in the working directory."))
}
print(paste("Loading data from:", combined_data_file))
# Handle potential parsing issues and new name warnings quietly for now
suppressMessages({
  combined_data <- readr::read_csv(combined_data_file, show_col_types = FALSE, name_repair = "minimal")
})
# Check for duplicate names explicitly if needed after loading
if (any(duplicated(names(combined_data)))){
  print("Warning: Duplicate column names found after loading. Consider cleaning the CSV file.")
  # Optionally rename duplicates here if necessary
}

df <- combined_data
print("Data loaded successfully.")


# --- Data Preprocessing ---
print("Starting data preprocessing...")

# Rename columns for easier use (Keep original renaming)
# Use tryCatch to handle potential errors if columns don't exist
tryCatch({
  df <- df %>%
    rename(
      ExperimentID = `experiment ID (participant number_condition)`,
      MRExperience_num = `Past exepriences with XR`, # Check spelling: experiences?
      ParticipantID = `participant number`,
      Condition_num = `condition`,
      RunOrder_num = `run order`,
      RPE = `RPE`,
      Satisfaction = `Satisfaction`,
      Enjoyment = `Enjoyment`,
      Challenging = `Challenging`,
      Concentration = `Concentration`,
      Presence = `Feeling of presence`, # Check for trailing space - NOTE: This original 'Presence' column might be different from the scale items
      MoveFreely = `Ability to move freely`,
      Fatigue = `Fatigue`,
      Exhaustion = `Exhaustion`,
      Sluggish = `Sluggish`,
      LightWeighted = `Feeling light weighted`,
      Relaxation = `Relaxation and losing oneself in the environment`, # PRS Item 1
      Curiosity = `Curiosity and fascination`, # PRS Item 2
      OrderInSpace = `Order in space`, # PRS Item 3
      EasyNavigation = `Easy navigation`, # PRS Item 4
      ComfortableEnv = `Comfortable environment`, # PRS Item 5
      Tense = `Tense`,
      Anxious = `Anxious`,
      Healthy = `Healthy`,
      Angry = `Angry`,
      Irritated = `Irritated`,
      Immersed = `Immersed`, # Presence Item 1
      PhysicallyPresent = `Physically present`, # Presence Item 2
      UseObjects = `Movement around and use of objects`, # Presence Item 3
      AbilityToDo = `Ability to do things`, # Presence Item 4, Check for trailing space
      Identification = `Identification`, # PA Item 1
      PerceivedGreenness = `Greenes of the environment`, # PA Item 2 (Env Perception) - NOTE: Not used in new PA index
      Belonging = `Sense of belonging`, # PA Item 3
      InterestingObserve = `Interesting things to observe`, # PA Item 4 (Env Perception) - NOTE: Not used in new PA index
      Monotonous = `Monotonous`, # PA Item 5 (Env Perception - Reverse?) - NOTE: Not used in new PA index
      NotMaintained = `Not well-maintained`, # PA Item 6 (Env Perception - Reverse?) - NOTE: Not used in new PA index
      Gender_num = `Gender`,
      Age = `Age`,
      Duration_char = `Duration`,
      StartTime_char = `Start time Watch`,
      TotalDistance_km_char = `Total distance (km)`,
      AvgHR_char = `Average heart rate (bpm)`,
      HR_max = `HR max`,
      AvgSpeed_kmh_char = `Average speed (km/h)`,
      MaxSpeed_kmh_char = `Max speed (km/h)`,
      AvgPace_char = `Average pace (min/km)`,
      MaxPace_char = `Max pace (min/km)`,
      Calories = `Calories`,
      AvgCadence = `Average cadence (rpm)`,
      AvgPower = `Average power (W)`,
      MaxPower = `Max power (W)`,
      Height_cm = `Height (cm)`,
      Weight_kg = `Weight (kg)`,
      VO2max = `VO2max`,
      Duration_s = `Duration_seconds`,
      Pace_minkm = `Recalculated_Pace_min_per_km`,
      Speed_kmh = `Recalculated_Speed_km_per_h`,
      MinHR = `min_heart_rate`,
      AvgSkinTemp = `avg_close_to_skin_temp`
    )
}, error = function(e){
  print(paste("Error during column renaming:", e$message))
  print("Check if all expected column names exist in the CSV file with the exact spelling/casing.")
})

# Convert data types & Create Factors
# Add checks to ensure columns exist before converting
if ("ParticipantID" %in% names(df)) df$ParticipantID <- factor(df$ParticipantID) else print("Warning: Column 'ParticipantID' not found.")
if ("Condition_num" %in% names(df)) df$Condition <- factor(df$Condition_num, levels = 0:2, labels = c("Neutral", "Shrubs", "Trees")) else print("Warning: Column 'Condition_num' not found.")
if ("RunOrder_num" %in% names(df)) df$RunOrder <- factor(df$RunOrder_num, levels = 1:2, labels = c("First", "Second")) else print("Warning: Column 'RunOrder_num' not found.")
if ("Gender_num" %in% names(df)) df$Gender <- factor(df$Gender_num, levels = 0:1, labels = c("Female", "Male")) else print("Warning: Column 'Gender_num' not found.")


# Fill MRExperience NAs and convert to factor
if ("MRExperience_num" %in% names(df) && "ParticipantID" %in% names(df)) {
  df <- df %>%
    group_by(ParticipantID) %>%
    fill(MRExperience_num, .direction = "downup") %>%
    ungroup()
  df$MRExperience <- factor(df$MRExperience_num, levels = 0:3, labels = c("None", "Few", "Some", "Many"))
  print("Initial distribution of MRExperience:")
  print(table(df$MRExperience, useNA = "ifany"))
} else {
  print("Warning: 'MRExperience_num' or 'ParticipantID' not found. Cannot process MRExperience.")
  df$MRExperience <- factor(NA) # Create placeholder
}


# Convert numerics, check HR_max
if ("AvgHR_char" %in% names(df)) df$AvgHR <- suppressWarnings(as.numeric(df$AvgHR_char)) else df$AvgHR <- NA
if ("HR_max" %in% names(df)) df$HR_max <- suppressWarnings(as.numeric(df$HR_max)) else df$HR_max <- NA
if ("MinHR" %in% names(df)) df$MinHR <- suppressWarnings(as.numeric(df$MinHR)) else df$MinHR <- NA
if ("AvgSkinTemp" %in% names(df)) df$AvgSkinTemp <- suppressWarnings(as.numeric(df$AvgSkinTemp)) else df$AvgSkinTemp <- NA


if("HR_max" %in% names(df) && any(is.na(df$HR_max)) || any(df$HR_max <= 0, na.rm=TRUE)) {
  warning("Missing or non-positive HR_max values found. %MaxHR calculation might produce NAs or errors.")
}

# Calculate % Max HR
if ("AvgHR" %in% names(df) && "HR_max" %in% names(df)) {
  df <- df %>%
    mutate(PercentMaxHR = ifelse(!is.na(HR_max) & HR_max > 0, (AvgHR / HR_max) * 100, NA))
} else {
  print("Warning: 'AvgHR' or 'HR_max' not available. Cannot calculate PercentMaxHR.")
  df$PercentMaxHR <- NA
}


# Create the Between-Subject 'Group' Factor (Used for descriptives/plots, but NOT in main model formula anymore)
if ("Condition" %in% names(df) && "ParticipantID" %in% names(df)) {
  participant_green_condition <- df %>%
    filter(Condition %in% c("Shrubs", "Trees")) %>% # Filter only green conditions
    distinct(ParticipantID, .keep_all = TRUE) %>%
    select(ParticipantID, Group = Condition) # Group is Shrubs or Trees
  df <- df %>% left_join(participant_green_condition, by = "ParticipantID")
  # Ensure Group is NA for Neutral condition explicitly
  df <- df %>% mutate(Group = if_else(Condition == "Neutral", factor(NA, levels=levels(participant_green_condition$Group)), Group))
  df$Group <- factor(df$Group) # Ensure it's a factor with levels Shrubs, Trees, NA
  print("Group factor created (Shrubs/Trees):")
  print(table(df$Group, useNA="ifany"))
} else {
  print("Warning: 'Condition' or 'ParticipantID' not available. Cannot create Group factor.")
  df$Group <- factor(NA)
}


# --- Create Time of Day Factor & Combine Categories ---
if ("StartTime_char" %in% names(df)) {
  df$StartTime_parsed <- suppressWarnings(lubridate::hms(df$StartTime_char))
  df$StartHour <- lubridate::hour(df$StartTime_parsed)
  df <- df %>%
    mutate(
      TimeOfDay_Initial = case_when(
        is.na(StartHour) ~ NA_character_,
        StartHour < 12 ~ "Morning",
        StartHour >= 12 & StartHour < 18 ~ "Afternoon",
        StartHour >= 18 ~ "Evening",
        TRUE ~ NA_character_
      )
    )
  print("Initial TimeOfDay distribution:")
  print(table(df$TimeOfDay_Initial, useNA = "ifany"))
  
  # Combine Afternoon and Evening
  df <- df %>%
    mutate(
      TimeOfDay = case_when(
        TimeOfDay_Initial == "Morning" ~ "Morning",
        TimeOfDay_Initial %in% c("Afternoon", "Evening") ~ "Afternoon/Evening", # Combine
        TRUE ~ NA_character_ # Keep NAs as NA
      )
    )
  # Convert TimeOfDay to a factor with the new levels
  df$TimeOfDay <- factor(df$TimeOfDay, levels = c("Morning", "Afternoon/Evening"))
  print("Created combined TimeOfDay factor (Morning, Afternoon/Evening).")
  print("Combined TimeOfDay distribution:")
  print(table(df$TimeOfDay, useNA = "ifany"))
  
} else {
  warning("Column 'Start time Watch' (renamed to StartTime_char) not found. Cannot create TimeOfDay factor.")
  df$TimeOfDay <- factor(NA, levels = c("Morning", "Afternoon/Evening")) # Create placeholder NA factor
}

# --- Reverse Code Items (If Needed - Note: Monotonous/NotMaintained removed from PA index) ---
# Keep this section in case other reversed items are needed later, but note the items are not used in PA.
if ("Monotonous" %in% names(df)) {
  df$Monotonous <- suppressWarnings(as.numeric(df$Monotonous))
  # Check if conversion worked before reversing
  if(is.numeric(df$Monotonous)) df$Monotonous_R <- 6 - df$Monotonous else df$Monotonous_R <- NA
}
if ("NotMaintained" %in% names(df)) {
  df$NotMaintained <- suppressWarnings(as.numeric(df$NotMaintained))
  if(is.numeric(df$NotMaintained)) df$NotMaintained_R <- 6 - df$NotMaintained else df$NotMaintained_R <- NA
}

# Define list of ORIGINAL potential dependent variables (before creating indices)
# We will add the indices later
original_dep_vars <- c("Duration_s", "AvgHR", "PercentMaxHR", "RPE",
                       "Enjoyment", "Satisfaction", "Fatigue", "Presence", # Note: 'Presence' here is the original single item, not the scale average yet
                       "Relaxation", "Curiosity", "Concentration",
                       "Tense", "Anxious", "Healthy", "Angry", "Irritated",
                       "Immersed", "PhysicallyPresent", "UseObjects", "AbilityToDo",
                       "PerceivedGreenness", # Keep for now, even if not in PA index
                       "MoveFreely", "Sluggish", "LightWeighted", "OrderInSpace", "EasyNavigation",
                       "ComfortableEnv", "Identification", "Belonging", "InterestingObserve", # Keep for now
                       "Monotonous", "NotMaintained", # Keep for now
                       "MinHR"
                       # Note: Reversed items Monotonous_R, NotMaintained_R are handled separately if needed.
)

# --- 3. Scale Reliability & Index Creation ---
print("Starting scale reliability checks and index creation...")

# Ensure all relevant columns are numeric for reliability checks and averaging
items_to_numeric <- c(
  # Presence items
  "Immersed", "PhysicallyPresent", "UseObjects", "AbilityToDo",
  # PRS items
  "Relaxation", "Curiosity", "OrderInSpace", "EasyNavigation", "ComfortableEnv",
  # PA items
  "Identification", "Belonging",
  # Positive Affect items
  "Enjoyment", "Satisfaction", "Healthy", "LightWeighted", # Relaxation already listed
  # Negative Affect items
  "Fatigue", "Tense", "Anxious", "Angry", "Irritated", "Sluggish"
)
# Only attempt conversion for columns that actually exist in df
items_to_numeric_present <- intersect(items_to_numeric, names(df))
if (length(items_to_numeric_present) > 0) {
  df <- df %>%
    mutate(across(all_of(items_to_numeric_present), ~ suppressWarnings(as.numeric(as.character(.)))))
} else {
  print("Warning: None of the expected item columns for scales found for numeric conversion.")
}


# --- Presence Scale ---
# NEW Items: Immersed, PhysicallyPresent, UseObjects, AbilityToDo
presence_item_names <- c("Immersed", "PhysicallyPresent", "UseObjects", "AbilityToDo")
# Check which of these items actually exist AND are numeric in df
presence_items_available <- intersect(presence_item_names, names(df))
presence_items_available <- presence_items_available[sapply(df[presence_items_available], is.numeric)]

alpha_presence_value <- NA
df$Presence_Avg <- NA # Initialize column

if(length(presence_items_available) == length(presence_item_names)) { # Only proceed if ALL expected items are available and numeric
  presence_items <- df %>% select(all_of(presence_items_available))
  presence_items_complete <- na.omit(presence_items)
  if(nrow(presence_items_complete) > 10 && ncol(presence_items_complete) == length(presence_items_available)) {
    print("--- Cronbach's Alpha for NEW Presence Scale Items ---")
    # Use tryCatch for psych::alpha as it can fail
    alpha_presence <- tryCatch({
      psych::alpha(presence_items_complete, check.keys=TRUE)
    }, error = function(e){ print(paste("psych::alpha failed:", e$message)); NULL })
    
    if (!is.null(alpha_presence)) {
      print(alpha_presence)
      alpha_presence_value <- alpha_presence$total$std.alpha
      if(!is.na(alpha_presence_value) && alpha_presence_value > 0.7) {
        print("Presence scale reliability acceptable. Creating average score 'Presence_Avg'.")
        df$Presence_Avg <- rowMeans(presence_items, na.rm = TRUE)
      } else {
        print("Presence scale reliability low or could not be calculated. Calculating 'Presence_Avg' based on available items anyway for covariate use.")
        df$Presence_Avg <- rowMeans(presence_items, na.rm = TRUE)
        df$Presence_Avg[rowSums(is.na(presence_items)) == ncol(presence_items)] <- NA
      }
    } else { # alpha calculation failed
      print("Cronbach's Alpha calculation failed. Calculating 'Presence_Avg' based on available items anyway for covariate use.")
      df$Presence_Avg <- rowMeans(presence_items, na.rm = TRUE)
      df$Presence_Avg[rowSums(is.na(presence_items)) == ncol(presence_items)] <- NA
    }
  } else {
    print("Could not calculate Cronbach's Alpha for Presence items (Insufficient complete cases or columns). Calculating row means anyway for covariate use.")
    df$Presence_Avg <- rowMeans(presence_items, na.rm = TRUE)
    df$Presence_Avg[rowSums(is.na(presence_items)) == ncol(presence_items)] <- NA
  }
} else {
  print(paste("One or more NEW Presence scale items not found or not numeric. Skipping Cronbach's Alpha. 'Presence_Avg' will be NA. Missing/non-numeric items:", paste(setdiff(presence_item_names, presence_items_available), collapse=", ")))
}


# --- Perceived Restorativeness Scale (PRS) ---
# NEW Items: Relaxation, Curiosity, OrderInSpace, EasyNavigation
prs_item_names <- c("Relaxation", "Curiosity", "OrderInSpace", "EasyNavigation")
prs_items_available <- intersect(prs_item_names, names(df))
prs_items_available <- prs_items_available[sapply(df[prs_items_available], is.numeric)]

alpha_prs_value <- NA
df$PRS_Avg <- NA # Initialize column

if(length(prs_items_available) == length(prs_item_names)) {
  prs_items <- df %>% select(all_of(prs_items_available))
  prs_items_complete <- na.omit(prs_items)
  if(nrow(prs_items_complete) > 10 && ncol(prs_items_complete) == length(prs_items_available)) {
    print("--- Cronbach's Alpha for NEW Perceived Restorativeness Scale Items ---")
    alpha_prs <- tryCatch({
      psych::alpha(prs_items_complete, check.keys=TRUE)
    }, error = function(e){ print(paste("psych::alpha failed:", e$message)); NULL })
    
    if (!is.null(alpha_prs)) {
      print(alpha_prs)
      alpha_prs_value <- alpha_prs$total$std.alpha
      if(!is.na(alpha_prs_value) && alpha_prs_value > 0.7) {
        print("PRS scale reliability acceptable. Creating average score 'PRS_Avg'.")
        df$PRS_Avg <- rowMeans(prs_items, na.rm = TRUE)
      } else {
        print("PRS scale reliability low or could not be calculated. Analyzing individual items or proceed with caution if using PRS_Avg.")
        df$PRS_Avg <- rowMeans(prs_items, na.rm = TRUE) # Calculate anyway?
        df$PRS_Avg[rowSums(is.na(prs_items)) == ncol(prs_items)] <- NA
      }
    } else {
      print("Cronbach's Alpha calculation failed. Calculating 'PRS_Avg' based on available items anyway.")
      df$PRS_Avg <- rowMeans(prs_items, na.rm = TRUE)
      df$PRS_Avg[rowSums(is.na(prs_items)) == ncol(prs_items)] <- NA
    }
  } else {
    print("Could not calculate Cronbach's Alpha for PRS items (Insufficient complete cases or columns). Calculating row means.")
    df$PRS_Avg <- rowMeans(prs_items, na.rm = TRUE)
    df$PRS_Avg[rowSums(is.na(prs_items)) == ncol(prs_items)] <- NA
  }
} else {
  print(paste("One or more NEW PRS items not found or not numeric. Skipping Cronbach's Alpha. 'PRS_Avg' will be NA. Missing/non-numeric items:", paste(setdiff(prs_item_names, prs_items_available), collapse=", ")))
}

# --- Place Attachment Scale ---
# NEW Items: Identification, Belonging
pa_item_names <- c("Identification", "Belonging")
pa_items_available <- intersect(pa_item_names, names(df))
pa_items_available <- pa_items_available[sapply(df[pa_items_available], is.numeric)]

alpha_pa_value <- NA # Alpha might be less meaningful with only 2 items
df$PlaceAttach_Avg <- NA # Initialize column

if(length(pa_items_available) == length(pa_item_names)) {
  pa_items <- df %>% select(all_of(pa_items_available))
  pa_items_complete <- na.omit(pa_items)
  if(nrow(pa_items_complete) > 10 && ncol(pa_items_complete) == length(pa_items_available)) {
    print("--- Cronbach's Alpha for NEW Place Attachment Items ---")
    alpha_pa <- tryCatch({
      psych::alpha(pa_items_complete, check.keys=TRUE)
    }, error = function(e){ print(paste("psych::alpha failed:", e$message)); NULL })
    
    if(!is.null(alpha_pa)){
      print(alpha_pa)
      alpha_pa_value <- alpha_pa$total$std.alpha
      # Adjusted threshold for 2 items, or just proceed based on correlation/theory
      if(!is.na(alpha_pa_value) && alpha_pa_value > 0.5) { # Lowered threshold example
        print("Place Attachment scale reliability potentially acceptable (check correlation too). Creating average score 'PlaceAttach_Avg'.")
        df$PlaceAttach_Avg <- rowMeans(pa_items, na.rm = TRUE)
      } else {
        print("Place Attachment scale reliability low or could not be calculated. Analyzing individual items or proceed with caution if using PlaceAttach_Avg.")
        df$PlaceAttach_Avg <- rowMeans(pa_items, na.rm = TRUE) # Calculate anyway?
        df$PlaceAttach_Avg[rowSums(is.na(pa_items)) == ncol(pa_items)] <- NA
      }
    } else {
      print("Cronbach's Alpha calculation failed. Calculating 'PlaceAttach_Avg' based on available items anyway.")
      df$PlaceAttach_Avg <- rowMeans(pa_items, na.rm = TRUE)
      df$PlaceAttach_Avg[rowSums(is.na(pa_items)) == ncol(pa_items)] <- NA
    }
  } else {
    print("Could not calculate Cronbach's Alpha for Place Attachment items (Insufficient complete cases or columns). Calculating row means.")
    df$PlaceAttach_Avg <- rowMeans(pa_items, na.rm = TRUE)
    df$PlaceAttach_Avg[rowSums(is.na(pa_items)) == ncol(pa_items)] <- NA
  }
} else {
  print(paste("One or more NEW Place Attachment items not found or not numeric. Skipping Cronbach's Alpha. 'PlaceAttach_Avg' will be NA. Missing/non-numeric items:", paste(setdiff(pa_item_names, pa_items_available), collapse=", ")))
}

# --- Positive Affect Scale ---
# NEW Items: Enjoyment, Satisfaction, Relaxation, Healthy, LightWeighted
pos_affect_item_names <- c("Enjoyment", "Satisfaction", "Relaxation", "Healthy", "LightWeighted")
pos_affect_items_available <- intersect(pos_affect_item_names, names(df))
pos_affect_items_available <- pos_affect_items_available[sapply(df[pos_affect_items_available], is.numeric)]

alpha_pos_affect_value <- NA
df$PositiveAffect_Avg <- NA # Initialize column

if(length(pos_affect_items_available) == length(pos_affect_item_names)) {
  pos_affect_items <- df %>% select(all_of(pos_affect_items_available))
  pos_affect_items_complete <- na.omit(pos_affect_items)
  if(nrow(pos_affect_items_complete) > 10 && ncol(pos_affect_items_complete) == length(pos_affect_items_available)) {
    print("--- Cronbach's Alpha for Positive Affect Items ---")
    alpha_pos_affect <- tryCatch({
      psych::alpha(pos_affect_items_complete, check.keys=TRUE)
    }, error = function(e){ print(paste("psych::alpha failed:", e$message)); NULL })
    
    if(!is.null(alpha_pos_affect)){
      print(alpha_pos_affect)
      alpha_pos_affect_value <- alpha_pos_affect$total$std.alpha
      if(!is.na(alpha_pos_affect_value) && alpha_pos_affect_value > 0.7) {
        print("Positive Affect scale reliability acceptable. Creating average score 'PositiveAffect_Avg'.")
        df$PositiveAffect_Avg <- rowMeans(pos_affect_items, na.rm = TRUE)
      } else {
        print("Positive Affect scale reliability low or could not be calculated. Proceed with caution if using PositiveAffect_Avg.")
        df$PositiveAffect_Avg <- rowMeans(pos_affect_items, na.rm = TRUE) # Calculate anyway?
        df$PositiveAffect_Avg[rowSums(is.na(pos_affect_items)) == ncol(pos_affect_items)] <- NA
      }
    } else {
      print("Cronbach's Alpha calculation failed. Calculating 'PositiveAffect_Avg' based on available items anyway.")
      df$PositiveAffect_Avg <- rowMeans(pos_affect_items, na.rm = TRUE)
      df$PositiveAffect_Avg[rowSums(is.na(pos_affect_items)) == ncol(pos_affect_items)] <- NA
    }
  } else {
    print("Could not calculate Cronbach's Alpha for Positive Affect items (Insufficient complete cases or columns). Calculating row means.")
    df$PositiveAffect_Avg <- rowMeans(pos_affect_items, na.rm = TRUE)
    df$PositiveAffect_Avg[rowSums(is.na(pos_affect_items)) == ncol(pos_affect_items)] <- NA
  }
} else {
  print(paste("One or more Positive Affect items not found or not numeric. Skipping Cronbach's Alpha. 'PositiveAffect_Avg' will be NA. Missing/non-numeric items:", paste(setdiff(pos_affect_item_names, pos_affect_items_available), collapse=", ")))
}

# --- Negative Affect Scale ---
# NEW Items: Fatigue, Tense, Anxious, Angry, Irritated, Sluggish
neg_affect_item_names <- c("Fatigue", "Tense", "Anxious", "Angry", "Irritated", "Sluggish")
neg_affect_items_available <- intersect(neg_affect_item_names, names(df))
neg_affect_items_available <- neg_affect_items_available[sapply(df[neg_affect_items_available], is.numeric)]

alpha_neg_affect_value <- NA
df$NegativeAffect_Avg <- NA # Initialize column

if(length(neg_affect_items_available) == length(neg_affect_item_names)) {
  neg_affect_items <- df %>% select(all_of(neg_affect_items_available))
  neg_affect_items_complete <- na.omit(neg_affect_items)
  if(nrow(neg_affect_items_complete) > 10 && ncol(neg_affect_items_complete) == length(neg_affect_items_available)) {
    print("--- Cronbach's Alpha for Negative Affect Items ---")
    alpha_neg_affect <- tryCatch({
      psych::alpha(neg_affect_items_complete, check.keys=TRUE)
    }, error = function(e){ print(paste("psych::alpha failed:", e$message)); NULL })
    
    if(!is.null(alpha_neg_affect)){
      print(alpha_neg_affect)
      alpha_neg_affect_value <- alpha_neg_affect$total$std.alpha
      if(!is.na(alpha_neg_affect_value) && alpha_neg_affect_value > 0.7) {
        print("Negative Affect scale reliability acceptable. Creating average score 'NegativeAffect_Avg'.")
        df$NegativeAffect_Avg <- rowMeans(neg_affect_items, na.rm = TRUE)
      } else {
        print("Negative Affect scale reliability low or could not be calculated. Proceed with caution if using NegativeAffect_Avg.")
        df$NegativeAffect_Avg <- rowMeans(neg_affect_items, na.rm = TRUE) # Calculate anyway?
        df$NegativeAffect_Avg[rowSums(is.na(neg_affect_items)) == ncol(neg_affect_items)] <- NA
      }
    } else {
      print("Cronbach's Alpha calculation failed. Calculating 'NegativeAffect_Avg' based on available items anyway.")
      df$NegativeAffect_Avg <- rowMeans(neg_affect_items, na.rm = TRUE)
      df$NegativeAffect_Avg[rowSums(is.na(neg_affect_items)) == ncol(neg_affect_items)] <- NA
    }
  } else {
    print("Could not calculate Cronbach's Alpha for Negative Affect items (Insufficient complete cases or columns). Calculating row means.")
    df$NegativeAffect_Avg <- rowMeans(neg_affect_items, na.rm = TRUE)
    df$NegativeAffect_Avg[rowSums(is.na(neg_affect_items)) == ncol(neg_affect_items)] <- NA
  }
} else {
  print(paste("One or more Negative Affect items not found or not numeric. Skipping Cronbach's Alpha. 'NegativeAffect_Avg' will be NA. Missing/non-numeric items:", paste(setdiff(neg_affect_item_names, neg_affect_items_available), collapse=", ")))
}
print("Scale reliability checks and index creation finished.")


# --- Define FINAL list of Dependent Variables ---
# Start with original list, add new indices, remove Presence_Avg
final_dep_vars <- unique(c(original_dep_vars, "PRS_Avg", "PlaceAttach_Avg", "PositiveAffect_Avg", "NegativeAffect_Avg"))
# Remove Presence_Avg as it's now a covariate
final_dep_vars <- setdiff(final_dep_vars, "Presence_Avg")
# Optional: Remove individual items that are now part of indices if desired
# items_in_indices <- c(presence_item_names, prs_item_names, pa_item_names, pos_affect_item_names, neg_affect_item_names)
# final_dep_vars <- setdiff(final_dep_vars, items_in_indices) # Uncomment this line to remove individual items from DV list

# --- Prepare Data Frame for Analysis ---
# Select necessary columns including new indices and Presence_Avg (for covariate)
all_needed_cols <- unique(c(
  "ParticipantID", "Condition", "Group", # Keep Group for plotting/descriptives
  "RunOrder", "Gender", "Age", "MRExperience",
  "VO2max", "HR_max", "AvgSkinTemp", "TimeOfDay", # Core IVs/Covariates
  "Presence_Avg", # ADDED Presence_Avg as potential covariate
  final_dep_vars # All DVs
))

# Ensure columns exist in df before selecting
all_needed_cols_exist <- all_needed_cols %in% names(df)
# --- ERROR FIX: Use any() to check if ANY needed columns are missing ---
if (any(!all_needed_cols_exist)) {
  missing_cols <- setdiff(all_needed_cols, names(df))
  print("Warning: The following columns needed for df_analysis are missing from df and will not be included:")
  print(missing_cols)
  all_needed_cols <- intersect(all_needed_cols, names(df)) # Keep only existing columns
}
# --- END ERROR FIX ---

# Select only the columns that actually exist in df
df_analysis <- df %>%
  select(all_of(intersect(all_needed_cols, names(df))))

# --- Relevel MRExperience Factor ---
if ("MRExperience" %in% names(df_analysis) && is.factor(df_analysis$MRExperience)) {
  if ("Some" %in% levels(df_analysis$MRExperience)) {
    print("Releveling MRExperience factor with 'Some' as the reference level.")
    df_analysis$MRExperience <- relevel(df_analysis$MRExperience, ref = "Some")
    print("New levels for MRExperience:")
    print(levels(df_analysis$MRExperience))
  } else {
    warning("Reference level 'Some' not found in MRExperience levels. Factor not releveled.")
  }
} else {
  warning("MRExperience column not found or not a factor in df_analysis. Cannot relevel.")
}

# Check which requested DVs are actually present in the final analysis dataframe
actual_dep_vars <- intersect(final_dep_vars, names(df_analysis))
missing_dep_vars <- setdiff(final_dep_vars, names(df_analysis))
if(length(missing_dep_vars) > 0) {
  print("Warning: The following requested dependent variables were not found in the final analysis dataframe and will be skipped:")
  print(missing_dep_vars)
}
final_dep_vars <- actual_dep_vars # Update list to only existing DVs
print("Data preprocessing finished.")
print(paste("Columns available for analysis in df_analysis:", paste(names(df_analysis), collapse=", ")))
print(paste("Dependent variables to be analyzed:", paste(final_dep_vars, collapse=", ")))


# --- 4. Mixed-Effects Models (LMMs) ---
print("Starting mixed-effects model fitting...")

# Define function to run model
run_lmm <- function(formula, data) {
  # Ensure DV is numeric before proceeding
  dv_name <- all.vars(formula)[1]
  if (!dv_name %in% names(data)) {
    print(paste("Error: Dependent variable", dv_name, "not found in data for formula:", deparse(formula)))
    return(NULL)
  }
  if (!is.numeric(data[[dv_name]])) {
    print(paste("Warning: DV", dv_name, "is not numeric. Attempting conversion."))
    data[[dv_name]] <- suppressWarnings(as.numeric(as.character(data[[dv_name]])))
    if (!is.numeric(data[[dv_name]])) {
      print(paste("Error: DV", dv_name, "could not be converted to numeric. Skipping model."))
      return(NULL)
    }
  }
  # Check for zero variance in DV
  # Also check for sufficient non-NA values
  valid_dv_data <- data[[dv_name]][!is.na(data[[dv_name]])]
  if(length(valid_dv_data) < 2) {
    print(paste("Dependent variable", dv_name, "has fewer than 2 non-NA values. Skipping LMM."))
    return(NULL)
  }
  # Use tryCatch for variance calculation in case of issues
  dv_variance <- tryCatch(var(valid_dv_data), error = function(e) NA)
  if(is.na(dv_variance) || dv_variance == 0) {
    print(paste("Dependent variable", dv_name, "has zero variance or variance could not be calculated after NA removal. Skipping LMM."))
    return(NULL)
  }
  
  # Proceed with original function logic
  all_vars_in_formula <- unique(c(all.vars(formula), all.vars(lme4::findbars(formula)[[1]])))
  # Check if all variables exist in the data
  missing_vars <- setdiff(all_vars_in_formula, names(data))
  if (length(missing_vars) > 0) {
    print(paste("Error: The following variables needed for the formula are missing from the data:", paste(missing_vars, collapse=", ")))
    return(NULL)
  }
  
  model_data_subset <- data[, intersect(names(data), all_vars_in_formula), drop = FALSE]
  model_data_complete <- na.omit(model_data_subset)
  n_obs <- nrow(model_data_complete)
  
  if (!"ParticipantID" %in% names(model_data_complete)) {
    print(paste("Error: ParticipantID missing after removing NAs for formula:", deparse(formula)))
    return(NULL)
  }
  # Check if ParticipantID has enough levels
  n_levels <- length(unique(model_data_complete$ParticipantID))
  if (n_levels < 2) {
    print(paste("Error: Fewer than 2 levels for ParticipantID after removing NAs for formula:", deparse(formula)))
    return(NULL)
  }
  
  # Estimate number of fixed effects carefully
  num_fixed_effects <- tryCatch({
    # Need to fit a temporary model just to get fixef length safely
    # Use suppressMessages to avoid printing temp model summary
    suppressMessages({
      # Ignore singular fit warnings for this temporary check
      temp_model <- lmer(formula, data=model_data_complete, control=lmerControl(calc.derivs=FALSE, check.nobs.vs.nRE = "ignore", check.nobs.vs.nlev = "ignore", check.conv.singular = .makeCC(action = "ignore")))
    })
    length(lme4::fixef(temp_model))
  }, error = function(e) {
    print(paste("Warning: Could not estimate number of fixed effects for pre-check:", e$message))
    # Guess based on formula terms (less reliable)
    length(all.vars(lme4::nobars(formula))) -1 # Rough estimate
  })
  if (is.null(num_fixed_effects)) num_fixed_effects <- 5 # Default guess if failed
  
  # Check observations vs parameters more carefully
  # n_ranef = number of random effects per participant (usually 1 for intercept only) * n_levels
  # This is approximate, actual degrees of freedom used might differ
  n_ranef_terms <- length(lme4::findbars(formula)[[1]]) # Number of random terms per level
  n_ranef_params <- n_levels * n_ranef_terms # Simplified estimate
  if (n_obs < num_fixed_effects + n_ranef_params) { # Basic check
    print(paste("Warning: Number of observations (", n_obs, ") might be low relative to fixed effects (", num_fixed_effects, ") and estimated random effects parameters for formula:", deparse(formula)))
  }
  if (n_obs <= n_levels) {
    print(paste("Warning: Number of observations (", n_obs, ") is less than or equal to the number of participants (", n_levels, ") for formula:", deparse(formula), ". Model likely unidentifiable."))
    # return(NULL) # Consider stopping here
  }
  
  
  print(paste("Fitting model with", n_obs, "observations for", n_levels, "participants."))
  model <- NULL
  diag_plot <- NULL # Initialize plot variable
  
  tryCatch({
    # Fit the actual model
    model <- lmer(formula, data = model_data_complete, REML = TRUE,
                  control=lmerControl(optimizer="bobyqa", # Try different optimizer if convergence issues
                                      calc.derivs=FALSE,
                                      check.nobs.vs.nRE = "warning", # Warn instead of error for potential issues
                                      check.nobs.vs.nlev = "warning"))
    print(summary(model))
    
    # Run diagnostics only if model fitting succeeded
    print("--- Running Model Diagnostics ---")
    # Use tryCatch for diagnostics as well, as they can sometimes fail
    diag_plot <- tryCatch({
      # performance::check_model can be memory intensive and fail on complex models
      # Consider simpler base R plots if check_model fails
      # Use panel=FALSE to generate a list of individual plots, potentially more robust
      performance::check_model(model, panel = FALSE)
    }, error = function(e_diag) {
      print(paste("Error generating diagnostic plots with performance::check_model:", e_diag$message))
      print("Attempting basic diagnostic plots...")
      tryCatch({
        # Ensure plotting device is available and clear previous plots
        if (dev.cur() == 1) dev.new() # Open a new device if none is open
        par(mfrow=c(2,2), mar=c(4,4,2,1), oma=c(0,0,2,0)) # Set margins
        plot(resid(model), fitted(model), main="Residuals vs Fitted", xlab="Fitted values", ylab="Residuals", pch=16, cex=0.8)
        abline(h=0, lty=2, col="grey")
        qqnorm(resid(model), main="Normal Q-Q Plot", pch=16, cex=0.8)
        qqline(resid(model), col="grey")
        plot(sqrt(abs(resid(model))), fitted(model), main="Scale-Location Plot", xlab="Fitted values", ylab=expression(sqrt(abs(Residuals))), pch=16, cex=0.8)
        # Optional: Add plot of random effects if simple (intercept only)
        # ranef_plot <- tryCatch(qqnorm(ranef(model)$ParticipantID[[1]], main="Random Intercept Q-Q"), error=function(e) NULL)
        # if(!is.null(ranef_plot)) qqline(ranef(model)$ParticipantID[[1]], col="grey")
        mtext("Basic Diagnostic Plots", outer=TRUE, cex=1.2)
        par(mfrow=c(1,1)) # Reset plot layout
      }, error = function(e_basic_plot){
        print(paste("Basic diagnostic plots also failed:", e_basic_plot$message))
      })
      NULL # Return NULL if plotting fails
    })
    # If diag_plot is a list of plots (from performance), print each one
    if (inherits(diag_plot, "list")) {
      # Need a graphics device open to print ggplot objects
      if (dev.cur() == 1) dev.new()
      # Print each plot in the list
      for(p in diag_plot) {
        tryCatch(print(p), error = function(e) print(paste("Could not print diagnostic plot:", e$message)))
      }
    } # Base plots are printed directly within the tryCatch block above
    
  }, error = function(e) {
    print(paste("Error fitting LMM:", e$message))
    # Check for specific common errors
    if (grepl("number of levels of grouping factor", e$message)) {
      print("This error often means the ParticipantID variable has too few levels after removing NAs for this specific model.")
    }
    if (grepl("Downdated VtV is not positive definite", e$message)) {
      print("This error might indicate collinearity or issues with the model structure/data.")
    }
    model <<- NULL # Ensure model is NULL on error
  }, warning = function(w) {
    print(paste("Warning during LMM fitting:", w$message))
    # Check for convergence warnings
    if (grepl("failed to converge", w$message, ignore.case = TRUE)) {
      print("Convergence warning detected. Results may be unreliable. Consider simplifying the model, changing optimizer, or checking scaling.")
      # Optionally treat convergence warning as an error: model <<- NULL
    }
    # Check for singular fit warnings
    if (grepl("singular fit", w$message, ignore.case = TRUE)) {
      print("Singular fit warning detected. This often means the random effects structure is too complex for the data (e.g., variance estimate near zero). Consider simplifying random effects.")
    }
  })
  
  return(model)
}


# Define covariates and model structure
# Ensure numeric covariates are numeric
numeric_covars <- c("Age", "VO2max", "AvgSkinTemp", "Presence_Avg") # Added Presence_Avg
numeric_covars_exist <- intersect(numeric_covars, names(df_analysis))
if (length(numeric_covars_exist) > 0) {
  df_analysis <- df_analysis %>%
    mutate(across(all_of(numeric_covars_exist), as.numeric))
}

# Ensure factor covariates are factors
factor_covars <- c("RunOrder", "Gender", "MRExperience", "TimeOfDay", "Condition") # Also ensure Condition is factor
factor_covars_exist <- intersect(factor_covars, names(df_analysis))
if (length(factor_covars_exist) > 0) {
  df_analysis <- df_analysis %>%
    mutate(across(all_of(factor_covars_exist), as.factor))
}

# Build the covariate string dynamically based on available columns
# Remove 'Condition' and 'Group' from base_covariates as they are main factors
base_covariates <- c("RunOrder", "Age", "Gender", "MRExperience", "VO2max", "AvgSkinTemp", "TimeOfDay", "Presence_Avg")
available_covariates <- intersect(base_covariates, names(df_analysis))

# --- ERROR FIX 2: Replace check_zeroinflation with manual variance check ---
# Check for zero variance in numeric covariates before adding them
numeric_available_covars <- available_covariates[sapply(df_analysis[, available_covariates, drop = FALSE], is.numeric)]
zero_var_covars <- c()
if (length(numeric_available_covars) > 0) {
  print("Checking numeric covariates for zero variance...")
  for (col in numeric_available_covars) {
    col_var <- tryCatch(var(df_analysis[[col]], na.rm = TRUE), error = function(e) NA)
    # Check if variance is NA (e.g., < 2 non-NA points) or exactly 0
    if (is.na(col_var) || (!is.na(col_var) && col_var == 0)) {
      zero_var_covars <- c(zero_var_covars, col)
    }
  }
  if (length(zero_var_covars) > 0) {
    print(paste("Warning: The following numeric covariates have zero variance or insufficient data and will be excluded:", paste(zero_var_covars, collapse=", ")))
    available_covariates <- setdiff(available_covariates, zero_var_covars)
  } else {
    print("No numeric covariates with zero variance found.")
  }
} else {
  print("No numeric covariates found to check for zero variance.")
}
# --- END ERROR FIX 2 ---


covariates_string <- ""
if (length(available_covariates) > 0) {
  covariates_string <- paste("+", paste(available_covariates, collapse = " + "))
}
print(paste("Using the following covariates in models:", covariates_string))

# --- UPDATED Model Structure ---
# Remove Group from the main model structure
model_structure <- " ~ Condition" # Condition has 3 levels: Neutral, Shrubs, Trees

# --- Fit models for FINAL list of dependent variables ---
model_list <- list()
print(paste("Analyzing DVs:", paste(final_dep_vars, collapse=", ")))

for (dv in final_dep_vars) {
  print(paste("--- Analyzing:", dv, "---"))
  if (!dv %in% names(df_analysis)) { print(paste(dv, "not found. Skipping.")); next }
  
  # Construct the formula string for the main model
  formula_string <- paste(dv, model_structure, covariates_string, "+ (1 | ParticipantID)")
  print(paste("Formula:", formula_string))
  
  # Convert to formula object
  formula_main <- tryCatch({
    as.formula(formula_string)
  }, error = function(e) {
    print(paste("Error creating formula for", dv, ":", e$message))
    NULL
  })
  
  if (is.null(formula_main)) { next } # Skip if formula creation failed
  
  # Run the LMM using the function
  model_list[[dv]] <- run_lmm(formula_main, df_analysis)
  
  # Simple check if model failed
  if (is.null(model_list[[dv]])) {
    print(paste("Model fitting failed for", dv))
  } else {
    print(paste("Model fitting potentially successful for", dv))
  }
}
print("Main model fitting finished.")


# --- 4.5 Specific Recovery Analysis (MinHR ~ Condition * RunOrder) ---
# Keep this section as it tests a specific interaction requested previously
# Note: This tests Condition (3 levels) * RunOrder (2 levels)
print("--- Specific Recovery Analysis for MinHR (Condition * RunOrder Interaction) ---")
recovery_model_minhr <- NULL
if ("MinHR" %in% final_dep_vars) { # Check if MinHR is still a DV to analyze
  # Construct the formula string including interaction and covariates
  recovery_formula_string <- paste("MinHR ~ Condition * RunOrder", covariates_string, "+ (1 | ParticipantID)")
  print(paste("Recovery Formula:", recovery_formula_string))
  
  recovery_formula <- tryCatch({
    as.formula(recovery_formula_string)
  }, error = function(e) {
    print(paste("Error creating recovery formula for MinHR:", e$message))
    NULL
  })
  
  if (!is.null(recovery_formula)) {
    print("Fitting MinHR model with Condition * RunOrder interaction...")
    recovery_model_minhr <- run_lmm(recovery_formula, df_analysis)
  }
} else {
  print("MinHR variable not found in the final list of DVs or analysis dataframe, skipping specific recovery analysis.")
}
print("MinHR recovery analysis finished.")


# --- 5. Post-Hoc Comparisons (Testing Hypotheses) ---
print("Starting post-hoc comparisons...")

# run_emmeans function (remains the same, but added checks)
run_emmeans <- function(model, specs, model_name = "model") {
  if (is.null(model)) { print(paste("Model for", model_name, "is NULL, skipping emmeans.")); return(NULL) }
  if (!inherits(model, "lmerMod")) { print(paste("Object for", model_name, "is not an lmerMod object. Skipping emmeans.")); return(NULL) }
  
  # Check if the terms in 'specs' actually exist in the model fixed effects or model frame
  spec_terms <- all.vars(specs)
  model_fixef_names <- tryCatch(names(fixef(model)), error = function(e) character(0)) # Handle cases where fixef fails
  model_frame_names <- tryCatch(names(model@frame), error = function(e) character(0)) # Get names from model data frame
  
  # Clean spec terms to handle interactions/simple terms
  clean_spec_terms <- unique(unlist(strsplit(gsub("~|\\||\\*|:", " ", deparse1(specs)), " "))) # Use deparse1 for robustness
  clean_spec_terms <- trimws(clean_spec_terms[clean_spec_terms != ""]) # Remove empty strings and trim whitespace
  
  term_exists_in_model <- sapply(clean_spec_terms, function(term) {
    term_found <- FALSE
    # Check if term matches exactly or partially in fixed effects (e.g., Condition for ConditionShrubs)
    if (length(model_fixef_names) > 0) {
      term_found <- term_found || any(grepl(term, model_fixef_names, fixed = FALSE)) # Use fixed=FALSE for partial match like Condition in ConditionShrubs
    }
    # Check if term exists as a column in the model frame data (important for factors)
    if (length(model_frame_names) > 0) {
      term_found <- term_found || (term %in% model_frame_names)
    }
    return(term_found)
  })
  
  
  if (!all(term_exists_in_model)) {
    missing_terms <- clean_spec_terms[!term_exists_in_model]
    print(paste("Warning: Term(s)", paste(missing_terms, collapse=", "), "in specs not found in the fixed effects or model frame for model", model_name, ". Skipping emmeans for:", deparse1(substitute(specs))))
    return(NULL)
  }
  
  
  emm_result <- NULL
  specs_char <- deparse1(substitute(specs)) # Use deparse1
  tryCatch({
    # Specify lmer.df="satterthwaite" or "kenward-roger" explicitly if desired and pbkrtest is available
    emm_result <- emmeans(model, specs = specs, lmer.df = "satterthwaite") # Requires pbkrtest to be loaded
    print(paste("--- Estimated Marginal Means for:", model_name, "by", specs_char, "---"))
    print(summary(emm_result)) # Show the means themselves
    print(paste("--- Pairwise Comparisons for:", model_name, "by", specs_char, "---"))
    print(pairs(emm_result))
    # Plotting can sometimes fail even if calculation works
    p <- tryCatch({
      plot(emm_result, comparisons = TRUE) + ggplot2::theme_minimal() + ggplot2::labs(title = paste("EMMeans Plot:", model_name, "by", specs_char))
    }, error = function(e_plot){
      print(paste("Could not generate emmeans plot:", e_plot$message))
      NULL
    })
    # Ensure graphics device is open before printing
    if (!is.null(p)) {
      if (dev.cur() == 1) dev.new()
      tryCatch(print(p), error = function(e) print(paste("Failed to print emmeans plot:", e$message)))
    }
  }, error = function(e) {
    print(paste("Could not run or display emmeans for", model_name, "with specs", specs_char, ":", e$message))
    # Check if error is due to degrees of freedom method
    if (grepl("lmer.df", e$message, ignore.case = TRUE)) {
      print("Trying emmeans without explicit d.f. method (using default)...")
      tryCatch({
        emm_result <- emmeans(model, specs = specs)
        print(paste("--- Estimated Marginal Means (Default DF):", model_name, "by", specs_char, "---"))
        print(summary(emm_result))
        print(paste("--- Pairwise Comparisons (Default DF):", model_name, "by", specs_char, "---"))
        print(pairs(emm_result))
        p <- tryCatch({
          plot(emm_result, comparisons = TRUE) + ggplot2::theme_minimal() + ggplot2::labs(title = paste("EMMeans Plot (Default DF):", model_name, "by", specs_char))
        }, error = function(e_plot){ NULL })
        if (!is.null(p)) {
          if (dev.cur() == 1) dev.new()
          tryCatch(print(p), error = function(e) print(paste("Failed to print fallback emmeans plot:", e$message)))
        }
      }, error = function(e_fallback){
        print(paste("Fallback emmeans also failed:", e_fallback$message))
        emm_result <<- NULL
      })
    } else {
      emm_result <<- NULL
    }
  })
  invisible(emm_result)
}

# Function to run Green vs Neutral contrast (remains the same)
run_green_contrast <- function(model, model_name_char) {
  if (is.null(model)) { print(paste("Model", model_name_char, "is NULL, skipping contrast.")); return(NULL) }
  if (!inherits(model, "lmerMod")) { print(paste("Object", model_name_char, "is not an lmerMod object. Skipping contrast.")); return(NULL) }
  
  # Check if 'Condition' is in the model frame data and is a factor
  if (!("Condition" %in% names(model@frame) && is.factor(model@frame$Condition))) {
    print(paste("Factor 'Condition' not found in the model frame for model", model_name_char, ". Skipping Green vs Neutral contrast."))
    return(NULL)
  }
  # Check if Condition has the expected levels
  condition_levels <- levels(model@frame$Condition) # Get levels from the model frame data
  if (!all(c("Neutral", "Shrubs", "Trees") %in% condition_levels)) {
    print(paste("Warning: Condition levels in model", model_name_char, "are not 'Neutral', 'Shrubs', 'Trees'. Green vs Neutral contrast might be incorrect or fail."))
    print("Levels found:")
    print(condition_levels)
    # Optional: return NULL here if levels are wrong
    # return(NULL)
  }
  
  
  emm_cond <- NULL
  tryCatch({ emm_cond <- emmeans(model, specs = ~ Condition, lmer.df = "satterthwaite") }, # Explicitly request Satterthwaite
           error = function(e) {
             print(paste("Could not calculate emmeans (Satterthwaite) for Green contrast on", model_name_char, ":", e$message))
             print("Trying emmeans for Green contrast with default d.f. method...")
             tryCatch({
               emm_cond <<- emmeans(model, specs = ~ Condition) # Fallback to default
             }, error = function(e_fallback){
               print(paste("Fallback emmeans for Green contrast also failed:", e_fallback$message))
               emm_cond <<- NULL
             })
           })
  
  if (is.null(emm_cond)) {
    print("Skipping Green contrast because emmeans calculation failed.")
    return(NULL)
  }
  
  # Ensure the contrast matches the levels order in emm_cond
  emm_levels <- tryCatch(levels(emm_cond[[1]]), error=function(e) NULL) # Safely get levels from the first component of emm grid
  if (is.null(emm_levels)) {
    print(paste("Warning: Could not retrieve levels from emmeans object for", model_name_char,". Green contrast might fail."))
    return(NULL)
  }
  
  contrast_list <- list(Green_vs_Neutral = c(-1, 0.5, 0.5)) # Default assuming Neutral, Shrubs, Trees order
  
  if (!identical(emm_levels, c("Neutral", "Shrubs", "Trees"))) {
    print(paste("Warning: Levels order in emmeans for", model_name_char, "is not 'Neutral', 'Shrubs', 'Trees'. Green contrast assumes this order."))
    print("Order found:")
    print(emm_levels)
    # Attempt to reorder contrast if possible (simple case)
    level_map <- match(c("Neutral", "Shrubs", "Trees"), emm_levels)
    if(!any(is.na(level_map)) && length(level_map) == 3) {
      ordered_contrast_vector <- numeric(3)
      ordered_contrast_vector[level_map] <- c(-1, 0.5, 0.5)
      contrast_list <- list(Green_vs_Neutral = ordered_contrast_vector)
      print("Green contrast vector reordered to match levels:")
      print(contrast_list)
    } else {
      print("Could not automatically reorder Green contrast vector. Proceeding with default, results may be incorrect.")
    }
  }
  
  
  print(paste("--- Contrast (Combined Green vs Neutral) for:", model_name_char, "---"))
  contrast_result <- NULL
  tryCatch({ contrast_result <- contrast(emm_cond, method = contrast_list); print(contrast_result) },
           error = function(e) { print(paste("Could not calculate Green contrast for", model_name_char, ":", e$message)); contrast_result <<- NULL })
  invisible(contrast_result)
}


# --- NEW: Function to run Shrubs vs Trees contrast ---
run_shrub_vs_tree_contrast <- function(model, model_name_char) {
  if (is.null(model)) { print(paste("Model", model_name_char, "is NULL, skipping Shrubs vs Trees contrast.")); return(NULL) }
  if (!inherits(model, "lmerMod")) { print(paste("Object", model_name_char, "is not an lmerMod object. Skipping contrast.")); return(NULL) }
  
  # Check if 'Condition' is in the model frame data and is a factor
  if (!("Condition" %in% names(model@frame) && is.factor(model@frame$Condition))) {
    print(paste("Factor 'Condition' not found in the model frame for model", model_name_char, ". Skipping Shrubs vs Trees contrast."))
    return(NULL)
  }
  # Check if Condition has the expected levels
  condition_levels <- levels(model@frame$Condition) # Get levels from the model frame data
  if (!all(c("Neutral", "Shrubs", "Trees") %in% condition_levels)) {
    print(paste("Warning: Condition levels in model", model_name_char, "are not 'Neutral', 'Shrubs', 'Trees'. Shrubs vs Trees contrast might be incorrect or fail."))
    print("Levels found:")
    print(condition_levels)
    # return(NULL) # Optional: stop if levels are wrong
  }
  
  emm_cond <- NULL
  tryCatch({ emm_cond <- emmeans(model, specs = ~ Condition, lmer.df = "satterthwaite") }, # Explicitly request Satterthwaite
           error = function(e) {
             print(paste("Could not calculate emmeans (Satterthwaite) for Shrubs vs Trees contrast on", model_name_char, ":", e$message))
             print("Trying emmeans for Shrubs vs Trees contrast with default d.f. method...")
             tryCatch({
               emm_cond <<- emmeans(model, specs = ~ Condition) # Fallback to default
             }, error = function(e_fallback){
               print(paste("Fallback emmeans for Shrubs vs Trees contrast also failed:", e_fallback$message))
               emm_cond <<- NULL
             })
           })
  
  if (is.null(emm_cond)) {
    print("Skipping Shrubs vs Trees contrast because emmeans calculation failed.")
    return(NULL)
  }
  
  # Ensure the contrast matches the levels order in emm_cond
  emm_levels <- tryCatch(levels(emm_cond[[1]]), error=function(e) NULL) # Safely get levels from the first component of emm grid
  if (is.null(emm_levels)) {
    print(paste("Warning: Could not retrieve levels from emmeans object for", model_name_char,". Shrubs vs Trees contrast might fail."))
    return(NULL)
  }
  
  # Contrast: Shrubs - Trees = 0
  contrast_list <- list(Shrubs_vs_Trees = c(0, 1, -1)) # Default assuming Neutral, Shrubs, Trees order
  
  if (!identical(emm_levels, c("Neutral", "Shrubs", "Trees"))) {
    print(paste("Warning: Levels order in emmeans for", model_name_char, "is not 'Neutral', 'Shrubs', 'Trees'. Shrubs vs Trees contrast assumes this order."))
    print("Order found:")
    print(emm_levels)
    # Attempt to reorder contrast
    level_map <- match(c("Neutral", "Shrubs", "Trees"), emm_levels)
    if(!any(is.na(level_map)) && length(level_map) == 3) {
      ordered_contrast_vector <- numeric(3)
      ordered_contrast_vector[level_map] <- c(0, 1, -1) # Reorder the 0, 1, -1 based on found order
      contrast_list <- list(Shrubs_vs_Trees = ordered_contrast_vector)
      print("Shrubs vs Trees contrast vector reordered to match levels:")
      print(contrast_list)
    } else {
      print("Could not automatically reorder Shrubs vs Trees contrast vector. Proceeding with default, results may be incorrect.")
    }
  }
  
  
  print(paste("--- Contrast (Shrubs vs Trees) for:", model_name_char, "---"))
  contrast_result <- NULL
  tryCatch({ contrast_result <- contrast(emm_cond, method = contrast_list); print(contrast_result) },
           error = function(e) { print(paste("Could not calculate Shrubs vs Trees contrast for", model_name_char, ":", e$message)); contrast_result <<- NULL })
  invisible(contrast_result)
}


# --- Run Post-Hoc tests for ALL models in model_list ---
posthoc_covariates <- c("RunOrder", "TimeOfDay", "MRExperience", "Gender", "Presence_Avg") # Covariates to potentially run post-hocs for (Removed Group)

for (dv in names(model_list)) {
  print(paste("--- Post-Hoc Tests for Main Model:", dv, "---"))
  current_model <- model_list[[dv]]
  if (!is.null(current_model) && inherits(current_model, "lmerMod")) {
    # Main effect of Condition (pairwise comparisons)
    run_emmeans(current_model, specs = ~ Condition, model_name = dv)
    # Contrast: Green (Avg) vs Neutral
    run_green_contrast(current_model, dv)
    # Contrast: Shrubs vs Trees (addresses between-group and interaction question)
    run_shrub_vs_tree_contrast(current_model, dv)
    
    # Post-hocs for covariates (only if they are factors with > 1 level)
    model_fixef_names <- tryCatch(names(fixef(current_model)), error = function(e) character(0))
    model_frame_names <- tryCatch(names(current_model@frame), error = function(e) character(0))
    
    for (covar in posthoc_covariates) {
      # Check if covariate is actually a factor in the model's data frame
      is_factor_in_frame <- (covar %in% model_frame_names) && is.factor(current_model@frame[[covar]])
      
      if(is_factor_in_frame) {
        # Only run emmeans if more than 1 level exists in the data used for the model
        model_levels <- tryCatch(levels(droplevels(current_model@frame[[covar]])), error = function(e) NULL)
        if (!is.null(model_levels) && length(model_levels) > 1) {
          run_emmeans(current_model, specs = as.formula(paste("~", covar)), model_name = paste(dv, covar, sep="_"))
        } else {
          # print(paste("Skipping emmeans for factor", covar, "in model", dv, "- only one level present in model data or could not get levels."))
        }
      } # Skip emmeans for numeric covariates
    }
    
  } else { print(paste("Skipping post-hoc for", dv, "as model fitting failed or object is not lmerMod.")) }
}

# --- Run Post-Hoc tests for the specific MinHR Recovery model ---
# Note: This model tests Condition * RunOrder interaction
print("--- Post-Hoc Tests for Specific MinHR Recovery Model (Condition * RunOrder) ---")
if (!is.null(recovery_model_minhr) && inherits(recovery_model_minhr, "lmerMod")) {
  print("Interaction: Condition by RunOrder")
  run_emmeans(recovery_model_minhr, specs = ~ Condition | RunOrder, model_name = "MinHR_Recovery_Cond_by_Run")
  print("Interaction: RunOrder by Condition")
  run_emmeans(recovery_model_minhr, specs = ~ RunOrder | Condition, model_name = "MinHR_Recovery_Run_by_Cond")
  
  # Post-hocs for covariates in the recovery model
  model_fixef_names_recov <- tryCatch(names(fixef(recovery_model_minhr)), error = function(e) character(0))
  model_frame_names_recov <- tryCatch(names(recovery_model_minhr@frame), error = function(e) character(0))
  
  for (covar in posthoc_covariates) { # Use the same list of potential covariates
    is_factor_in_frame_recov <- (covar %in% model_frame_names_recov) && is.factor(recovery_model_minhr@frame[[covar]])
    
    if(is_factor_in_frame_recov) {
      model_levels_recov <- tryCatch(levels(droplevels(recovery_model_minhr@frame[[covar]])), error = function(e) NULL)
      if (!is.null(model_levels_recov) && length(model_levels_recov) > 1) {
        run_emmeans(recovery_model_minhr, specs = as.formula(paste("~", covar)), model_name = paste("MinHR_Recovery", covar, sep="_"))
      } # Skip if only one level
    } # Skip numeric covariates
  }
  
} else { print("Skipping post-hoc for MinHR recovery model as fitting failed or object is not lmerMod.") }
print("Post-hoc comparisons finished.")


# --- 5.5 ANOVA/ANCOVA Style Tables from Mixed Models ---
print("Generating ANOVA tables...")

# Function to safely run and print ANOVA
run_anova <- function(model, model_name) {
  if (!is.null(model) && inherits(model, "lmerMod")) {
    print(paste("--- ANOVA Table (Type III Satterthwaite) for Model:", model_name, "---"))
    tryCatch({
      # Use lmerTest::anova
      anova_table <- anova(model, type = 3, ddf = "Satterthwaite")
      print(anova_table)
    }, error = function(e) {
      print(paste("Could not compute Satterthwaite ANOVA table for", model_name, ":", e$message))
      # Try Kenward-Roger as fallback? Requires pbkrtest
      if (requireNamespace("pbkrtest", quietly = TRUE)) {
        print("Trying ANOVA with Kenward-Roger ddf...")
        tryCatch({
          anova_table_kr <- anova(model, type = 3, ddf = "Kenward-Roger")
          print(anova_table_kr)
        }, error = function(e_kr) {
          print(paste("Kenward-Roger ANOVA also failed for", model_name, ":", e_kr$message))
          # Final fallback: Wald chi-square tests (Type III) from car::Anova
          if(requireNamespace("car", quietly = TRUE)) {
            print("Trying ANOVA with Wald chi-square tests (car::Anova)...")
            tryCatch({
              # Need to specify the test statistic for car::Anova with lmer
              anova_table_car <- car::Anova(model, type = 3, test.statistic="Chisq")
              print(anova_table_car)
            }, error = function(e_car){
              print(paste("car::Anova also failed for", model_name, ":", e_car$message))
            })
          }
        })
      } else if(requireNamespace("car", quietly = TRUE)) {
        print("pbkrtest not found. Trying ANOVA with Wald chi-square tests (car::Anova)...")
        tryCatch({
          anova_table_car <- car::Anova(model, type = 3, test.statistic="Chisq")
          print(anova_table_car)
        }, error = function(e_car){
          print(paste("car::Anova also failed for", model_name, ":", e_car$message))
        })
      } else {
        print("Neither pbkrtest nor car packages seem available for fallback ANOVA methods.")
      }
    })
  } else {
    print(paste("Skipping ANOVA for", model_name, "as model is NULL or not an lmerMod object."))
  }
}

# --- ANOVA Tables for All Main Models ---
print("--- ANOVA Tables for All Main Models ---")
# Ensure car package is loaded if needed for fallback ANOVA
if (!requireNamespace("car", quietly = TRUE)) {
  print("Loading car package for potential fallback ANOVA...")
  library(car)
}
for (dv in names(model_list)) {
  run_anova(model_list[[dv]], dv)
}

# --- ANOVA Table for Specific MinHR Recovery Model ---
print("--- ANOVA Table for Specific MinHR Recovery Model (Condition * RunOrder) ---")
run_anova(recovery_model_minhr, "MinHR_Recovery")

print("ANOVA table generation finished.")


# --- 6. Visualizations ---
print("Generating visualizations (will be included in output file)...")

# Define key variables to plot, including new indices
key_vars_plot <- intersect(c("Duration_s", "AvgHR", "PercentMaxHR", "RPE",
                             "PositiveAffect_Avg", # New
                             "NegativeAffect_Avg", # New
                             "PRS_Avg",            # Updated
                             "PlaceAttach_Avg",    # Updated
                             "MinHR", "AvgSkinTemp"
                             # Presence_Avg is now a covariate, maybe less interesting to plot this way?
),
names(df_analysis)) # Ensure they exist in the analysis data

print("--- Generating Box Plots for Key Variables ---")
for (dv in key_vars_plot) {
  if (!dv %in% names(df_analysis)) { print(paste("Skipping plot for", dv, "- not found.")); next }
  if (!is.numeric(df_analysis[[dv]])) { print(paste("Skipping plot for", dv, "- not numeric.")); next }
  # Check if DV has variance before plotting
  # Also check for sufficient non-NA points
  valid_dv_plot_data <- df_analysis[[dv]][!is.na(df_analysis[[dv]])]
  if(length(valid_dv_plot_data) < 2) {
    print(paste("Skipping plot for", dv, "- < 2 non-NA values."))
    next
  }
  dv_plot_variance <- tryCatch(var(valid_dv_plot_data), error = function(e) NA)
  if(is.na(dv_plot_variance) || dv_plot_variance == 0) {
    print(paste("Skipping plot for", dv, "- zero variance or variance could not be calculated."))
    next
  }
  
  
  print(paste("Plotting Boxplot for:", dv))
  
  # Plot by Condition, faceting by RunOrder, using Group for fill
  # Ensure Group exists for plotting
  if ("Group" %in% names(df_analysis) && "Condition" %in% names(df_analysis) && "RunOrder" %in% names(df_analysis)) {
    plot_data_cond <- df_analysis %>% filter(!is.na(Condition) & !is.na(RunOrder) & !is.na(.data[[dv]]))
    # Add Group info back for coloring, filter out Neutral for Group coloring logic if needed
    plot_data_cond <- plot_data_cond %>% mutate(FillGroup = ifelse(Condition=="Neutral", "Neutral", as.character(Group)))
    plot_data_cond$FillGroup <- factor(plot_data_cond$FillGroup, levels=c("Neutral", "Shrubs", "Trees")) # Ensure order
    
    if(nrow(plot_data_cond) > 0) {
      # Suppress warnings during plotting (e.g., removed rows with NA)
      suppressWarnings({
        p_cond <- ggplot(plot_data_cond, aes(x = Condition, y = .data[[dv]], fill = FillGroup)) +
          geom_boxplot(position = position_dodge(0.8), alpha = 0.7, na.rm = TRUE, outlier.shape = NA) + # Hide boxplot outliers if adding points
          geom_point(position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8), alpha = 0.3, size=0.8, na.rm = TRUE) + # Add points
          facet_wrap(~RunOrder) +
          scale_fill_brewer(palette = "Pastel1", name = "Condition/Group", na.value="grey80") + # Updated legend name
          labs(title = paste(dv, "by Condition (Faceted by Run Order)"),
               x = "Environmental Condition", y = dv) +
          theme_minimal(base_size = 10) + # Adjust base size if needed
          theme(legend.position = "bottom", axis.text.x = element_text(angle = 45, hjust = 1))
        if (dev.cur() == 1) dev.new()
        tryCatch(print(p_cond), error = function(e) print(paste("Failed to print Condition boxplot:", e$message)))
      })
    } else { print(paste("No data to plot for", dv, "by Condition/RunOrder after removing NAs.")) }
  } else { print("Skipping Condition boxplot - Condition, Group, or RunOrder column missing.")}
  
  
  # Plot by TimeOfDay (Combined)
  if ("TimeOfDay" %in% names(df_analysis)) {
    plot_data_time <- df_analysis %>% filter(!is.na(TimeOfDay) & !is.na(.data[[dv]]))
    if(nrow(plot_data_time) > 0 && length(unique(na.omit(plot_data_time$TimeOfDay))) > 1) { # Ensure >1 level for plotting
      suppressWarnings({
        p_time <- ggplot(plot_data_time, aes(x = TimeOfDay, y = .data[[dv]], fill = TimeOfDay)) +
          geom_boxplot(alpha = 0.7, na.rm = TRUE, show.legend = FALSE, outlier.shape = NA) + # Hide outliers if adding points
          geom_jitter(width = 0.2, alpha = 0.4, size=1, na.rm=TRUE, show.legend = FALSE) + # Add jittered points
          scale_fill_brewer(palette = "Set2", na.value="grey80") +
          labs(title = paste(dv, "by Time of Day (Combined)"), x = "Time of Day", y = dv) +
          theme_minimal(base_size = 10)
        if (dev.cur() == 1) dev.new()
        tryCatch(print(p_time), error = function(e) print(paste("Failed to print TimeOfDay boxplot:", e$message)))
      })
    } else { print(paste("No data or only one level to plot for", dv, "by TimeOfDay after removing NAs.")) }
  }
}

# Interaction plot (emmip) for key variables from MAIN models
print("--- Generating EMMeans Plots (emmip) for Key Variables (Main Models) ---")
# Function to safely create emmip plots
create_emmip <- function(model, formula, title_suffix) {
  if (is.null(model) || !inherits(model, "lmerMod")) return(NULL)
  
  # Check if terms exist before plotting
  terms_in_formula <- all.vars(formula)
  model_fixef_names <- tryCatch(names(fixef(model)), error = function(e) character(0))
  model_frame_names <- tryCatch(names(model@frame), error = function(e) character(0))
  
  term_exists_in_model <- sapply(terms_in_formula, function(term) {
    term_found <- FALSE
    if (length(model_fixef_names) > 0) {
      # Check if term exists exactly or as part of an interaction/factor level name
      term_found <- term_found || any(grepl(paste0("\\b", term, "\\b"), model_fixef_names)) || any(startsWith(model_fixef_names, term))
    }
    if (length(model_frame_names) > 0) {
      term_found <- term_found || (term %in% model_frame_names)
    }
    return(term_found)
  })
  
  if (!all(term_exists_in_model)) {
    missing_terms <- terms_in_formula[!term_exists_in_model]
    print(paste("Skipping emmip for", title_suffix, "- term(s)", paste(missing_terms, collapse=", "), "not found in model."))
    return(NULL)
  }
  
  # Check if factor levels allow plotting
  for(term in terms_in_formula) {
    if(term %in% model_frame_names && is.factor(model@frame[[term]])) {
      model_levels <- tryCatch(levels(droplevels(model@frame[[term]])), error = function(e) NULL)
      if(is.null(model_levels) || length(model_levels) < 2) {
        print(paste("Skipping emmip for", title_suffix, "- factor", term, "has less than 2 levels in model data or levels could not be determined."))
        return(NULL)
      }
    }
  }
  
  plot_obj <- NULL # Initialize plot object
  tryCatch({
    # Use lmer.df = "satterthwaite" if pbkrtest is available and desired
    plot_obj <- emmip(model, formula, lmer.df = "satterthwaite") + ggplot2::labs(title = paste("EMMeans:", title_suffix)) + ggplot2::theme_minimal(base_size=10)
  }, error = function(e){
    print(paste("Could not create emmip plot for", title_suffix, "(Satterthwaite):", e$message))
    # Fallback without explicit df method
    print("Trying emmip plot with default df method...")
    tryCatch({
      plot_obj <<- emmip(model, formula) + ggplot2::labs(title = paste("EMMeans (Default DF):", title_suffix)) + ggplot2::theme_minimal(base_size=10)
    }, error = function(e_fallback){
      print(paste("Fallback emmip plot also failed:", e_fallback$message))
      plot_obj <<- NULL # Ensure it's NULL on fallback failure too
    })
  })
  return(plot_obj) # Return the plot object (or NULL)
}

for (dv in key_vars_plot) {
  if (!dv %in% names(model_list) || is.null(model_list[[dv]])) { print(paste("Skipping emmip for", dv, "- model not found or invalid.")); next }
  print(paste("Plotting emmip for:", dv))
  
  # Plot main effect of Condition
  p_emmip_cond <- create_emmip(model_list[[dv]], ~ Condition, paste(dv, "by Condition"))
  if(!is.null(p_emmip_cond)) {
    if (dev.cur() == 1) dev.new()
    tryCatch(print(p_emmip_cond), error = function(e) print(paste("Failed to print Condition emmip plot:", e$message)))
  }
  
  # Remove Group emmip plot as Group is not in the main model
  # p_emmip_group <- create_emmip(model_list[[dv]], ~ Group, paste(dv, "by Group"))
  # if(!is.null(p_emmip_group)) print(p_emmip_group)
  
  # Plot main effect of RunOrder (if in model)
  p_emmip_run <- create_emmip(model_list[[dv]], ~ RunOrder, paste(dv, "by Run Order"))
  if(!is.null(p_emmip_run)) {
    if (dev.cur() == 1) dev.new()
    tryCatch(print(p_emmip_run), error = function(e) print(paste("Failed to print RunOrder emmip plot:", e$message)))
  }
  
  # Plot main effect of TimeOfDay (if in model)
  p_emmip_time <- create_emmip(model_list[[dv]], ~ TimeOfDay, paste(dv, "by Time of Day"))
  if(!is.null(p_emmip_time)) {
    if (dev.cur() == 1) dev.new()
    tryCatch(print(p_emmip_time), error = function(e) print(paste("Failed to print TimeOfDay emmip plot:", e$message)))
  }
  
  # Plot main effect of MRExperience (if in model)
  p_emmip_mr <- create_emmip(model_list[[dv]], ~ MRExperience, paste(dv, "by MR Experience"))
  if(!is.null(p_emmip_mr)) {
    if (dev.cur() == 1) dev.new()
    tryCatch(print(p_emmip_mr), error = function(e) print(paste("Failed to print MRExperience emmip plot:", e$message)))
  }
  
}

# Interaction plot (emmip) specifically for the MinHR Recovery model interaction (Condition * RunOrder)
print("--- Generating EMMeans Plot (emmip) for MinHR Recovery Interaction (Condition * RunOrder) ---")
if (!is.null(recovery_model_minhr) && inherits(recovery_model_minhr, "lmerMod")) {
  p_emmip_recov <- create_emmip(recovery_model_minhr, Condition ~ RunOrder, "MinHR Interaction (Condition * RunOrder)")
  if (!is.null(p_emmip_recov)) {
    if (dev.cur() == 1) dev.new()
    tryCatch(print(p_emmip_recov), error = function(e) print(paste("Failed to print MinHR recovery emmip plot:", e$message)))
  }
} else { print("Skipping emmip for MinHR recovery model - model not found or invalid.")}
print("Visualization generation finished.")


# --- 7. Explanation of Results Guidance (Updated for Mixed Design Contrasts) ---
print("--- Interpretation Guidance (Mixed Design via Contrasts) ---")
print("# --- Key Changes Made ---")
print("# - Main model formula is now: dv ~ Condition + Covariates + (1|ParticipantID), where Condition has 3 levels (Neutral, Shrubs, Trees).")
print("# - 'Group' factor removed from the main model formula.")
print("# - Analysis addresses Within (Control vs Green), Between (Shrubs vs Trees), and Interaction effects using contrasts on the main model's 'Condition' factor.")
print("# - Presence_Avg is included as a covariate.")
print("# - TimeOfDay combines Afternoon/Evening.")
print("# - MRExperience uses 'Some' as reference.")
print("# --- Interpretation Steps ---")
print("# 1. Review Cronbach's Alpha values (printed earlier) for PRS_Avg, PlaceAttach_Avg, PositiveAffect_Avg, NegativeAffect_Avg. Decide if average scores are reliable.")
print("# 2. Review ANOVA tables (Type III) for the main models:")
print("#    - Significant `Condition` effect? This indicates an overall difference among the three levels (Neutral, Shrubs, Trees). Check specific contrasts/post-hocs below.")
print("#    - Significant COVARIATE effects (e.g., `Presence_Avg`, `TimeOfDay`, `MRExperience`, `Age`, `Gender`, `VO2max`, `AvgSkinTemp`)?")
print("#      - For FACTOR covariates (RunOrder, TimeOfDay, MRExperience, Gender): Check post-hoc (`run_emmeans(..., specs = ~ CovariateName)`) if significant in ANOVA.")
print("#      - For NUMERIC covariates (Presence_Avg, Age, VO2max, AvgSkinTemp): Interpret the coefficient sign and significance from the `summary(model)` output.")
print("# 3. Review Post-Hoc Contrasts for the 'Condition' effect from the main model:")
print("#    - `Green_vs_Neutral` contrast: Tests the WITHIN-SUBJECT effect (Control vs. Average Green). Is there a significant difference between Neutral and the average of Shrubs/Trees?")
print("#    - `Shrubs_vs_Trees` contrast: Tests the BETWEEN-SUBJECT difference *within* the green conditions. Do Shrubs and Trees significantly differ from each other?")
print("#    - INTERACTION interpretation: The significance of the `Shrubs_vs_Trees` contrast ALSO informs the interaction. If Shrubs significantly differ from Trees, it implies that the effect of Greenery (compared to Neutral) DEPENDS on the type of greenery (Shrubs vs. Trees). If the contrast is non-significant, there's no evidence the type of green matters.")
print("#    - Pairwise comparisons from `run_emmeans(..., specs = ~ Condition)` show all direct comparisons (Neutral vs Shrubs, Neutral vs Trees, Shrubs vs Trees).")
print("# 4. Review the separate ANOVA table for the specific MinHR Recovery model:")
print("#    - This model tests the `Condition:RunOrder` interaction specifically for MinHR. Interpret this interaction if significant using its specific post-hoc tests.")
print("# 5. Check diagnostic plots (printed during model fitting) for potential model assumption violations.")
print("# 6. Review Box Plots and EMMeans plots (emmip) to visualize significant effects and contrasts.")


# --- STOP: Restore Console Output ---
print(paste("Analysis finished at:", Sys.time()))
print(paste("Full output saved to:", output_filename))
sink() # Stop redirecting output to the file
# --- END: Restore Console Output ---

# --- End of Script ---
print("Script finished. Check the output file for complete results:")
print(output_filename)

