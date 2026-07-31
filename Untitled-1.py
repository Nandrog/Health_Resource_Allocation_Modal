# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# %%

df = pd.read_csv('Health_Resource_Allocation_Dataset.csv')
print("first five rows of the dataset")
print(df.head())
print('-'*50)



# %%
print("Information about the dataset")
df.info()
print('-'*50)
print("Statistical summary of the dataset")
df.describe()

# %%
numerical_columns = ['Population', 'Doctors', 'Hospital_Beds', 'Budget_Million_UGX']

for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# %%
# Check missing values and % missing per feature
missing_counts = df.isna().sum()
missing_pct = (df.isna().mean() * 100).round(2)

missing_summary = pd.DataFrame({
    'missing_count': missing_counts,
    'missing_%': missing_pct
}).sort_values('missing_%', ascending=False)

print(missing_summary)

# %%
fig, ax = plt.subplots(figsize=(9, 5))
missing_summary['missing_%'].sort_values(ascending=False).plot.bar(ax=ax)
ax.set_title('Missing Data Percentage by Feature')
ax.set_ylabel('% missing')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
#impute missing values using mean 
cols_to_impute = ['Population', 'Doctors', 'Hospital_Beds', 'Budget_Million_UGX', 'Medicine_Stock']

df_before_impute = df[cols_to_impute].copy()

for col in cols_to_impute:
    if df[col].isna().sum() > 0:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
        print(f"{col}: filled {missing_counts[col]} missing value(s) with mean = {mean_val}")

# %%
# Confirm no missing values remain, and that the fill didn't distort the distribution
print("Missing values remaining:\n", df.isna().sum())

print("\nUpdated statistical summary:\n", df[cols_to_impute].describe())



# %%
fig, axes = plt.subplots(1, len(cols_to_impute), figsize=(18, 4))
for ax, col in zip(axes, cols_to_impute):
    sns.histplot(df_before_impute[col].dropna(), kde=True, ax=ax, color='steelblue')
    ax.set_title(col)
plt.suptitle('Distribution BEFORE Imputation')
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, len(cols_to_impute), figsize=(18, 4))
for ax, col in zip(axes, cols_to_impute):
    sns.histplot(df[col], kde=True, ax=ax, color='steelblue')
    ax.set_title(col)
plt.suptitle('Distribution AFTER Imputation')
plt.tight_layout()
plt.show()

# %%
#A barplot for districts against population
plt.figure(figsize=(10, 6))
sns.barplot(x='District', y='Population', data=df, errorbar=None, palette='viridis')
plt.title('Population Distribution by District')
plt.xlabel('District')
plt.ylabel('Population')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
#distribution analysis for doctors, hospital beds, budget, and medicine stock
for col in cols_to_impute:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# %%
#comparing resource allocation across districts using barplots for doctors, hospital beds, budget, and medicine stock
for col in cols_to_impute:
    plt.figure(figsize=(10, 6))
    #exclude population distribution as it has already been plotted
    if col != 'Population':
        sns.barplot(x='District', y=col, data=df, errorbar=None, palette='coolwarm')
        plt.title(f'{col} Distribution by District')
        plt.xlabel('District')
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# %%
plt.figure(figsize=(10, 8))
corr_matrix = df[cols_to_impute].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap: Population vs Resources')
plt.tight_layout()
plt.show()

# %%
#outlier detection using boxplots for doctors, hospital beds, budget, and medicine stock
for col in cols_to_impute:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# %%

#Sorting - find the districts with the highest disease cases

df.sort_values("Disease_Cases", ascending=False).head(10)

# %%

# Lowest medicine stock - could signal districts at risk of shortages
df.sort_values("Medicine_Stock").head(10)

# %%
# The 6 "noise" columns we're checking
noise_cols = ['Medicine_Stock', 'Population', 'Budget_Million_UGX',
              'Hospital_Beds', 'Health_Facilities', 'Nurses']

# Create a grid of 6 small charts (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()  # turns the 2x3 grid into a simple list of 6 slots

# Loop through each noisy column and draw its own bar chart
for i, col in enumerate(noise_cols):
    avg_by_need = df.groupby('Resource_Need_Level')[col].mean()
    avg_by_need = avg_by_need.reindex(['Low', 'Medium', 'High'])
    
    axes[i].bar(avg_by_need.index, avg_by_need.values, color=['gray', 'darkgray', 'black'])
    axes[i].set_title(col)

plt.tight_layout()  # stops the charts from overlapping each other
plt.show()

# %%
#  Average Doctors per need level
avg_doctors = df.groupby('Resource_Need_Level')['Doctors'].mean()

# Put it in logical order (Low, Medium, High)
avg_doctors = avg_doctors.reindex(['Low', 'Medium', 'High'])

print(avg_doctors)

# Plot it
plt.bar(avg_doctors.index, avg_doctors.values, color=['lightgreen', 'green', 'darkgreen'])
plt.title('Average Doctors by Resource Need Level')
plt.xlabel('Resource Need Level')
plt.ylabel('Average Number of Doctors')
plt.show()

# %%
#  Calculate the average Disease_Cases for each need level
avg_disease = df.groupby('Resource_Need_Level')['Disease_Cases'].mean()

#  Reorder so it goes Low -> Medium -> High (otherwise it's alphabetical)
avg_disease = avg_disease.reindex(['Low', 'Medium', 'High'])

print(avg_disease)

# Step 3: Plot it as a bar chart
plt.bar(avg_disease.index, avg_disease.values, color=['lightblue', 'blue', 'darkblue'])
plt.title('Average Disease Cases by Resource Need Level')
plt.xlabel('Resource Need Level')
plt.ylabel('Average Disease Cases')
plt.show()

# %%
import numpy as np

# 1. Fix Impossible Values: Negative Doctors
print(f"Number of rows with negative doctors BEFORE: {len(df[df['Doctors'] < 0])}")

# We will replace any number of doctors less than 0 with the median (middle) value
median_doctors = df['Doctors'].median()
df.loc[df['Doctors'] < 0, 'Doctors'] = median_doctors

print(f"Number of rows with negative doctors AFTER: {len(df[df['Doctors'] < 0])}")

# 2. Fix the Extreme Outlier: Log Transformation (Keeping Original!)
print("\nCreating new Log-Transformed Medicine_Stock column...")

# Create the new column with the logged values
df['Medicine_Stock_Log'] = np.log1p(df['Medicine_Stock'])


print("Medicine_Stock_Log successfully created! Original column kept.")

# 3. Final sanity check on our fixes
# Let's look at the original and the logged version side-by-side
display(df[['Doctors', 'Medicine_Stock', 'Medicine_Stock_Log']].describe())

# %%

cols_to_impute.append('Medicine_Stock_Log')
for col in cols_to_impute:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

print(cols_to_impute)


# %%


# ==========================================
# PART 1: CHECKING FOR NEGATIVE VALUES
# ==========================================
print("--- Checking all numeric columns for negative values ---")

# Automatically select only the columns that are numbers (integers or floats)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Check if values are less than 0, then sum them up per column
negative_counts = (df[numeric_cols] < 0).sum()

# Filter to only show columns that have more than 0 negative values
negatives_found = negative_counts[negative_counts > 0]

if len(negatives_found) == 0:
    print("Great news! No negative values found in any numeric column.")
else:
    print("Warning! Negative values found in:")
    display(negatives_found)

print("\n" + "="*42 + "\n")

# ==========================================
# PART 2: ENCODING (KEEPING ORIGINALS)
# ==========================================
print("--- Encoding Features ---")

# Target Encoding: Low=0, Medium=1, High=2 (preserves real order, unlike LabelEncoder)
need_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['Target_Encoded'] = df['Resource_Need_Level'].map(need_mapping)

# One-Hot Encoding: Region only (District excluded - doesn't generalize to new districts)
dummies = pd.get_dummies(df[['Region']], drop_first=True)

# Glue (concatenate) the original dataframe and the new dummy columns side-by-side (axis=1)
df_encoded = pd.concat([df, dummies], axis=1)

print("Original columns kept! Features encoded successfully.")
print(f"New Dataset Shape: {df_encoded.shape}")

# Let's look at the columns to prove the original and new ones are both there
display(df_encoded[['District', 'Region', 'Resource_Need_Level', 'Target_Encoded']].head(10))

# %%
df.head(10)

# %%
#feature selection
selected_features = ['Health_Facilities', 'Doctors', 'Population', 'Nurses', 'Hospital_Beds', 'Disease_Cases', 'Medicine_Stock_Log']
x = df[selected_features]
y = df['Target_Encoded']

#train, test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state=42)

# Create the scaler
scaler = StandardScaler()

# Fit on training data and transform it
x_train = scaler.fit_transform(x_train)

# Transform the test data using the same scaler
x_test = scaler.transform(x_test)


# %%
#random forest classifier 
# we aim at classifying under resource need levels

model = RandomForestClassifier(
    n_estimators=100,        # Number of trees in the forest
    max_depth=10,            # How deep each tree can grow
    min_samples_split=5,     # Minimum samples required to split a node
    min_samples_leaf=2,      # Minimum samples required in a leaf node
    random_state=42)

print("Train random forest classifier")

model.fit(x_train, y_train)

print("Model testing....")
y_pred =model.predict(x_test)

# %%
#evaluation
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy on Test Set: {accuracy:.2%}")

unique_classes = sorted(y.unique())
class_names = [f'Level_{cls}' for cls in unique_classes]  # Example: Level_1, Level_2, Level_3

# Show detailed statistics
print("\n Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Show the Confusion Matrix (how many correct vs wrong)
print("\n Confusion Matrix (Rows=Actual, Columns=Predicted):")
print(confusion_matrix(y_test, y_pred))


# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, 
                                      display_labels=class_names,
                                      cmap='Blues',
                                      ax=ax)
ax.set_title('🌳 Random Forest Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 6. THE BEST PART: FEATURE IMPORTANCE!
# ------------------------------------------------------------
# This tells you which columns the model relied on the most.
import matplotlib.pyplot as plt

importances = model.feature_importances_
feature_names = selected_features

# Create a nice bar chart
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='forestgreen')
plt.xlabel('Feature Importance')
plt.ylabel('feature_names')
plt.title(' What factors predicted survival the most?')
plt.tight_layout()
plt.show()

# Print the exact numbers
print("\n Feature Importance Scores:")
for name, score in zip(feature_names, importances):
    print(f"   - {name}: {score:.2%}")




