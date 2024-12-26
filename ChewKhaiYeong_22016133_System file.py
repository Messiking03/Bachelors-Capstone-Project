import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score
#Simulated Data Code
#Simulated Dataset Creation

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for normal distributions
sample_size = 1000
age_mean, age_std = 40, 12  # Mean age 40, std dev 12
height_mean, height_std = 170, 10  # Mean height 170 cm, std dev 10
weight_mean, weight_std = 70, 15  # Mean weight 70 kg, std dev 15

# Generate normally distributed variables
age = np.random.normal(age_mean, age_std, sample_size).astype(int)  # Normally distributed Age
height = np.random.normal(height_mean, height_std, sample_size).astype(int)  # Normally distributed Height
weight = np.random.normal(weight_mean, weight_std, sample_size).astype(int)  # Normally distributed Weight

# Generate skewed variables
income = (np.random.exponential(scale=50000, size=sample_size) + 20000).astype(int)  # Right-skewed Income
exercise_hours = (np.random.exponential(scale=3, size=sample_size)).round(0).astype(int)  # Right-skewed Exercise Hours

# Clip the values to keep within reasonable bounds
age = np.clip(age, 18, 70)  # Age between 18 and 70
height = np.clip(height, 150, 200)  # Height between 150 and 200 cm
weight = np.clip(weight, 50, 120)  # Weight between 50 and 120 kg
income = np.clip(income, 20000, 150000)  # Income between 20k and 150k USD
exercise_hours = np.clip(exercise_hours, 0, 15)  # Exercise hours between 0 and 15

# Create the DataFrame
simulated_df = pd.DataFrame({
    'X1': age,
    'X2': income,
    'X3': height,
    'X4': weight,
    'X5': exercise_hours
})

# Function to introduce missing values based on MAR (Missing At Random) conditions
def introduce_mar_missingness_sim(simulated_df, missing_percentage=0.50):
    np.random.seed(0)  # For reproducibility
    n_rows, n_cols = simulated_df.shape
    n_missing = int(n_rows * n_cols * missing_percentage)
    
    # Define the conditions for MAR (we don't affect 'Age' directly)
    median_income = simulated_df['X2'].median()
    median_weight = simulated_df['X4'].median()
    
    conditions = {
        'X2': simulated_df['X1'] < 30,  # Introduce missingness to 'Income' where 'Age' < 30
        'X4': simulated_df['X1'] > 50,  # Introduce missingness to 'Weight' where 'Age' > 50
        'X5': simulated_df['X4'] > median_weight  # Introduce missingness to 'Exercise Hours' where 'Weight' > median
    }

    for col, condition in conditions.items():
        indices = simulated_df[condition].index
        n_missing_col = min(len(indices), int(n_missing / len(conditions)))
        
        # Randomly select indices to introduce missingness
        if n_missing_col > 0:
            missing_indices = np.random.choice(indices, n_missing_col, replace=False)
            simulated_df.loc[missing_indices, col] = np.nan
    
    return simulated_df

# Introduce missing values using the defined function
df_with_missing_values_sim = introduce_mar_missingness_sim(simulated_df, missing_percentage=0.50)


# Create a heatmap to visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df_with_missing_values_sim.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()

#Mean Imputation
mean_imputed_sim = df_with_missing_values_sim.copy()  # Create a copy to preserve the original dataframe

# Replace missing values with the mean of each respective column
mean_imputed_sim['X2'].fillna(mean_imputed_sim['X2'].mean(), inplace=True)
mean_imputed_sim['X4'].fillna(mean_imputed_sim['X4'].mean(), inplace=True)
mean_imputed_sim['X5'].fillna(mean_imputed_sim['X5'].mean(), inplace=True)
mean_imputed_sim['X5'] = mean_imputed_sim['X5'].round(0).astype(int)  # Round to whole numbers
# Print the dataframe after imputation
print(mean_imputed_sim.head())

# visualize imputed data to verify the absence of missing data
plt.figure(figsize=(10, 6))
sns.heatmap(mean_imputed_sim.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()


#KNN Imputation

# Create a heatmap to visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df_with_missing_values_sim.isnull(), cbar=False, cmap='coolwarm', yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()



knn_imputer_sim = KNNImputer(n_neighbors=5)
knn_imputed_sim = pd.DataFrame(knn_imputer_sim.fit_transform(df_with_missing_values_sim), columns=df_with_missing_values_sim.columns)
knn_imputed_sim['X5'] = knn_imputed_sim['X5'].round(0).astype(int)  # Round to whole numbers

# visualize imputed data to verify the absence of missing data
plt.figure(figsize=(10, 6))
sns.heatmap(knn_imputed_sim.isnull(), cbar=False, cmap='coolwarm', yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()

#Regression Imputation
# Create a heatmap to visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df_with_missing_values_sim.isnull(), cbar=False, cmap='YlOrRd', yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()


#  (for 'Income' and 'Weight')
regression_imputed_sim = df_with_missing_values_sim.copy()  # Create a copy of the original DataFrame

# 1. For 'Income' using 'Age' and 'Height' as predictors
train_income = regression_imputed_sim.dropna(subset=['X2'])[['X1', 'X3']]  # Training data (rows where 'Income' is not missing)
target_income = regression_imputed_sim.dropna(subset=['X2'])['X2']  # Target variable for training (non-missing 'Income' values)
test_income = regression_imputed_sim[regression_imputed_sim['X2'].isna()][['X1', 'X3']]  # Test data (rows where 'Income' is missing)

if not test_income.empty:  # If there are missing values in 'Income'
    reg = LinearRegression()  # Initialize linear regression model
    reg.fit(train_income, target_income)  # Fit the model to the non-missing 'Income' data
    regression_imputed_sim.loc[regression_imputed_sim['X2'].isna(), 'X2'] = reg.predict(test_income)  # Predict missing 'Income' values and fill them in the DataFrame


# 2. For 'Weight' using 'Age' and 'Height' as predictors
train_weight = regression_imputed_sim.dropna(subset=['X4'])[['X1', 'X3']]  # Training data (rows where 'Weight' is not missing)
target_weight = regression_imputed_sim.dropna(subset=['X4'])['X4']  # Target variable for training (non-missing 'Weight' values)
test_weight = regression_imputed_sim[regression_imputed_sim['X4'].isna()][['X1', 'X3']]  # Test data (rows where 'Weight' is missing)

if not test_weight.empty:  # If there are missing values in 'Weight'
    reg = LinearRegression()  # Initialize linear regression model
    reg.fit(train_weight, target_weight)  # Fit the model to the non-missing 'Weight' data
    regression_imputed_sim.loc[regression_imputed_sim['X4'].isna(), 'X4'] = np.round(reg.predict(test_weight)).astype(int)



# 3. For 'Exercise Hours' using 'Weight' as a predictor
train_exercise = regression_imputed_sim.dropna(subset=['X5'])[['X4']]  # Training data (rows where 'Exercise Hours' is not missing)
target_exercise = regression_imputed_sim.dropna(subset=['X5'])['X5']  # Target variable for training (non-missing 'Exercise Hours' values)
test_exercise = regression_imputed_sim[regression_imputed_sim['X5'].isna()][['X4']]  # Test data (rows where 'Exercise Hours' is missing)

if not test_exercise.empty:  # If there are missing values in 'Exercise Hours'
    reg = LinearRegression()  # Initialize linear regression model
    reg.fit(train_exercise, target_exercise)  # Fit the model to the non-missing 'Exercise Hours' data
    regression_imputed_sim.loc[regression_imputed_sim['X5'].isna(), 'X5'] = reg.predict(test_exercise)  # Predict missing 'Exercise Hours' values and fill them in the DataFrame

# Round 'Exercise Hours' in regression imputation to whole numbers
regression_imputed_sim['EX5'] = regression_imputed_sim['X5'].round(0).astype(int)  # Ensure 'Exercise Hours' is rounded to whole numbers


# Visualize imputed data to verify the absence of missing data
plt.figure(figsize=(10, 6))
sns.heatmap(regression_imputed_sim.isnull(), cbar=False, cmap='YlOrRd', yticklabels=False)  # 'YlOrRd' is a red-orange colormap
plt.title("Missing Data Heatmap")
plt.show()

#Perform Agglomerative Hierarchical Clustering for each imputation
# Ensure all features exist in the imputed DataFrames
mean_imputed_sim = mean_imputed_sim.copy()
knn_imputed = knn_imputed_sim.copy()
regression_imputed_sim = regression_imputed_sim.copy()

features_to_cluster = ['X1', 'X2', 'X3', 'X4', 'X5']

#Create DataFrames for clustering
mean_df_sim = mean_imputed_sim[features_to_cluster].dropna()
knn_df_sim = knn_imputed_sim[features_to_cluster].dropna()
regression_df_sim = regression_imputed_sim[features_to_cluster].dropna()

# Define the number of clusters
n_clusters_sim = 3

# Fit and predict clusters for each imputed dataset
mean_clustering_sim = AgglomerativeClustering(n_clusters=n_clusters_sim)
knn_clustering_sim = AgglomerativeClustering(n_clusters=n_clusters_sim)
regression_clustering_sim = AgglomerativeClustering(n_clusters=n_clusters_sim)


mean_labels_sim = mean_clustering_sim.fit_predict(mean_df_sim)
knn_labels_sim = knn_clustering_sim.fit_predict(knn_df_sim)
regression_labels_sim = regression_clustering_sim.fit_predict(regression_df_sim)

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Function to plot dendrograms for each dataset
def plot_dendrograms_sim(data_list_sim, titles_sim):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Create a row of 3 plots

    for i, (data, title) in enumerate(zip(data_list_sim, titles_sim)):
        # Perform hierarchical clustering (linkage) for dendrogram
        linked = linkage(data, method='complete')  # You can choose other methods like 'average', 'complete', etc.

        # Plot the dendrogram
        dendrogram(linked, ax=axes[i], truncate_mode='lastp', p=20, leaf_rotation=45, leaf_font_size=10)
        axes[i].set_title(title)
        axes[i].set_xlabel('Sample index or (cluster size)')
        axes[i].set_ylabel('Distance')

    plt.tight_layout()
    plt.show()

# Create a list of data and titles for each plot
data_list_sim = [mean_imputed_sim, knn_imputed_sim, regression_imputed_sim]
titles_sim = ['Dendrogram for Mean Imputed Data', 'Dendrogram for KNN Imputed Data', 'Dendrogram for Regression Imputed Data']

# Call the function to create a combined dendrogram plot
plot_dendrograms_sim(data_list_sim, titles_sim)

#Produce the output of performance of each imputation method for clustering
# Function to perform agglomerative clustering and evaluate with silhouette and DBI
def evaluate_clustering_sim(data, n_clusters, method_name):
    # Perform Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters_sim)
    labels = clustering.fit_predict(data)
    
    # Compute Silhouette Score
    silhouette_avg_sim = silhouette_score(data, labels)
    
    # Compute Davies-Bouldin Index
    dbi_score_sim = davies_bouldin_score(data, labels)
    

    return labels

# Function to calculate Dunn Index without scaling
def dunn_index_sim(data_sim, cluster_labels_sim):
    # Intra-cluster distances (compactness)
    intra_distances_sim = []
    for cluster_sim in set(cluster_labels_sim):
        cluster_points_sim = data_sim[cluster_labels_sim == cluster_sim]
        if len(cluster_points_sim) > 1:  # Ensure more than one point in the cluster
            pairwise_distances_sim = np.linalg.norm(cluster_points_sim[:, np.newaxis] - cluster_points_sim, axis=2)
            max_intra_distance_sim = np.max(pairwise_distances_sim)  # Max distance within the cluster
            intra_distances_sim.append(max_intra_distance_sim)
        else:
            intra_distances_sim.append(0)  # For singleton clusters, intra-cluster distance is 0
    max_intra_sim = np.max(intra_distances_sim)  # Largest intra-cluster distance across all clusters

    # Inter-cluster distances (separation)
    inter_distances_sim = []
    unique_clusters_sim = list(set(cluster_labels_sim))
    for i, cluster_i_sim in enumerate(unique_clusters_sim):
        for j, cluster_j_sim in enumerate(unique_clusters_sim):
            if i < j:  # Avoid duplicate calculations
                points_i_sim = data_sim[cluster_labels_sim == cluster_i_sim]
                points_j_sim = data_sim[cluster_labels_sim == cluster_j_sim]
                inter_dist_sim = np.min(np.linalg.norm(points_i_sim[:, np.newaxis] - points_j_sim, axis=2))
                inter_distances_sim.append(inter_dist_sim)
    min_inter_sim = np.min(inter_distances_sim)  # Smallest inter-cluster distance

    # Calculate Dunn Index
    dunn_index_value_sim = min_inter_sim / max_intra_sim if max_intra_sim > 0 else float('inf')  # Avoid division by zero
    return dunn_index_value_sim


# Initialize DataFrame to store clustering evaluation results
results_df_sim = pd.DataFrame(columns=['Imputation Method', 'Silhouette Score', 'Davies-Bouldin Index', 'Dunn Index'])

# Perform clustering evaluations for each imputed dataset
# 1. Mean Imputation
mean_labels_sim = evaluate_clustering_sim(mean_df_sim, n_clusters_sim, 'Mean Imputation')
mean_dunn_index_sim = dunn_index_sim(mean_df_sim.values, mean_labels_sim)
results_df_sim = results_df_sim.append({
    'Imputation Method': 'Mean Imputation',
    'Silhouette Score': silhouette_score(mean_df_sim, mean_labels_sim),
    'Davies-Bouldin Index': davies_bouldin_score(mean_df_sim, mean_labels_sim),
    'Dunn Index': mean_dunn_index_sim
}, ignore_index=True)

# 2. KNN Imputation
knn_labels_sim = evaluate_clustering_sim(knn_df_sim, n_clusters_sim, 'KNN Imputation')
knn_dunn_index_sim = dunn_index_sim(knn_df_sim.values, knn_labels_sim)
results_df_sim = results_df_sim.append({
    'Imputation Method': 'KNN Imputation',
    'Silhouette Score': silhouette_score(knn_df_sim, knn_labels_sim),
    'Davies-Bouldin Index': davies_bouldin_score(knn_df_sim, knn_labels_sim),
    'Dunn Index': knn_dunn_index_sim
}, ignore_index=True)

# 3. Regression Imputation
regression_labels_sim = evaluate_clustering_sim(regression_df_sim, n_clusters_sim, 'Regression Imputation')
regression_dunn_index_sim = dunn_index_sim(regression_df_sim.values, regression_labels_sim)
results_df_sim = results_df_sim.append({
    'Imputation Method': 'Regression Imputation',
    'Silhouette Score': silhouette_score(regression_df_sim, regression_labels_sim),
    'Davies-Bouldin Index': davies_bouldin_score(regression_df_sim, regression_labels_sim),
    'Dunn Index': regression_dunn_index_sim
}, ignore_index=True)

# Display the results DataFrame
print(results_df_sim)


# Extract metrics from results DataFrame for plotting
methods_sim = results_df_sim['Imputation Method']
silhouette_scores_sim = results_df_sim['Silhouette Score']
davies_bouldin_indices_sim = results_df_sim['Davies-Bouldin Index']
dunn_indices_sim = results_df_sim['Dunn Index']

# Plotting
x = np.arange(len(methods_sim))  # label locations
width = 0.25  # bar width

fig, ax = plt.subplots(figsize=(12, 6))

# Create bar plots for Silhouette Score, Davies-Bouldin Index, and Dunn Index
bars1 = ax.bar(x - width, silhouette_scores_sim, width, label='Silhouette Score', color='b')
bars2 = ax.bar(x, davies_bouldin_indices_sim, width, label='Davies-Bouldin Index', color='r')
bars3 = ax.bar(x + width, dunn_indices_sim, width, label='Dunn Index', color='g')

# Labels and title
ax.set_ylabel('Scores')
ax.set_title('Clustering Evaluation Metrics by Imputation Method')
ax.set_xticks(x)
ax.set_xticklabels(methods_sim)
ax.legend()

# Add value annotations on the bars
def add_value_labels_sim(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_value_labels_sim(bars1)
add_value_labels_sim(bars2)
add_value_labels_sim(bars3)

plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage


#Real Data Code
# Load the dataset
df_real = pd.read_csv(r"C:\Users\khaiy\OneDrive\Documents\Capstone Project 2\ChewKhaiYeong_22016133_heart_disease_dataset.csv")  # Replace with your file path

# Get basic information of dataset
print(df_real.info())

# View the first few rows
print(df_real.head())

# Path to your output Excel file
output_path = r"C:\Users\khaiy\OneDrive\Documents\Capstone Project 2\Summ_Stats.xlsx"

# Summary statistics for numerical variables
numerical_summary = df_real.describe()

# Summary statistics for categorical variables
categorical_summary = df_real.describe(include=['object'])

# Print numerical and categorical summaries
print("Numerical Summary:")
print(numerical_summary)

print("\nCategorical Summary:")
print(categorical_summary)

# Save both summaries to the Excel file
with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
    # Save numerical summary to the first sheet
    numerical_summary.to_excel(writer, sheet_name='SummaryStats')
    
    # Save categorical summary to the second sheet
    categorical_summary.to_excel(writer, sheet_name='SummaryCatStats')

print(f"Numerical and categorical summary statistics saved to: {output_path}")


# Heatmap for missing values
plt.figure(figsize=(10,6))
sns.heatmap(df_real.isnull(), cbar=False, cmap='viridis')
plt.title("Heatmap of Realdata")
plt.show()

#EDA before introducing missing values
#Numerical Variable Explorations
# Plot Distribution and Histogram for the numerical columns
# Determine the number of numerical columns
selected_columns_1 = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Exercise Hours', 'Stress Level', 'Blood Sugar']
num_plots = len(selected_columns_1)

# Define the number of rows and columns for the subplot grid
# Adjust rows and columns based on how many columns you have for better visualization
n_cols = 3  # Number of columns per row
n_rows = (num_plots // n_cols) + (num_plots % n_cols > 0)  # Calculate number of rows needed

# Set up the subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
fig.suptitle("Distributions of Numerical Columns", fontsize=16, y=1.02)

# Flatten axes array if necessary (for easy iteration)
axes = axes.flatten()

# Plot each numerical column's distribution
for i, column in enumerate(selected_columns_1):
    sns.histplot(df_real[column], kde=True, bins=30, ax=axes[i])
    axes[i].set_title(f"Distribution of {column}")
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Frequency")

# Remove any empty subplots if there are extra grid spaces
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# List of categorical columns you want to visualize
categorical_columns_1 = ['Gender', 'Smoking', 'Alcohol Intake', 'Family History', 'Diabetes', 'Obesity', 'Exercise Induced Angina', 'Chest Pain Type', 'Heart Disease']

# Determine the number of plots
num_plots = len(categorical_columns_1)

# Define the number of rows and columns for the subplot grid
n_cols = 3  # Number of columns per row
n_rows = (num_plots // n_cols) + (num_plots % n_cols > 0)  # Calculate the number of rows needed

# Set up the subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
fig.suptitle("Count Plots of Categorical Variables", fontsize=16, y=1.02)

# Flatten axes array if necessary (for easy iteration)
axes = axes.flatten()

# Plot each categorical column's count plot
for i, column in enumerate(categorical_columns_1):
    sns.countplot(x=column, data=df_real, ax=axes[i])
    axes[i].set_title(f"Count Plot of {column}")
    axes[i].set_ylabel("Count")

# Remove any empty subplots if there are extra grid spaces
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Correlation matrix
corr_matrix = df_real.corr()

# Heatmap for correlation
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Manually select columns
selected_columns_1 = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Exercise Hours', 'Stress Level', 'Blood Sugar']   

# Perform One-hot encode categorical variables
df_real_encoded = pd.get_dummies(df_real, columns=[
    'Gender', 'Diabetes', 'Smoking', 'Alcohol Intake', 
    'Family History', 'Obesity', 'Exercise Induced Angina', 
    'Chest Pain Type', 'Exercise Hours'], drop_first=True)

# Step 2: Function to introduce MAR missing values with 50% missing percentage
def introduce_mar(df_real_encoded):
    # Introduce missing values for numerical variables
    # Cholesterol missing for Age > 50
    chol_condition = df_real_encoded['Age'] > 50
    chol_indices = df_real_encoded[chol_condition].sample(frac=0.5, random_state=60).index
    df_real_encoded.loc[chol_indices, 'Cholesterol'] = np.nan
    
    # Blood Pressure missing for Male (assuming Male is represented as 'Gender_Male')
    bp_condition = df_real_encoded['Gender_Male'] == 1
    bp_indices = df_real_encoded[bp_condition].sample(frac=0.5, random_state=42).index
    df_real_encoded.loc[bp_indices, 'Blood Pressure'] = np.nan
    
    # Blood Sugar missing for those without diabetes (assuming Diabetes is represented as 'Diabetes_Yes')
    bs_condition = df_real_encoded['Diabetes_Yes'] == 0
    bs_indices = df_real_encoded[bs_condition].sample(frac=0.5, random_state=50).index
    df_real_encoded.loc[bs_indices, 'Blood Sugar'] = np.nan

    # Heart Rate missing for those who do not exercise (assuming at least one 'Exercise Hours' column)
    exercise_columns = [col for col in df_real_encoded.columns if 'Exercise Hours_' in col]
    if exercise_columns:
        hr_condition = df_real_encoded[exercise_columns].sum(axis=1) == 0
        hr_indices = df_real_encoded[hr_condition].sample(frac=0.5, random_state=42).index
        df_real_encoded.loc[hr_indices, 'Heart Rate'] = np.nan
        
    # NEW: Introduce MAR missing values randomly in the 'Age' variable (50% of rows)
    age_indices = df_real_encoded.sample(frac=0.5, random_state=70).index
    df_real_encoded.loc[age_indices, 'Age'] = np.nan  # Introduce missing values for Age
    
    # Introduce missing values in categorical variables
    # Example: Introduce missing values in 'Gender' for individuals with Diabetes (using one-hot encoded columns)
    condition_diabetes = df_real_encoded['Diabetes_Yes'] == 1
    diabetes_indices = df_real_encoded[condition_diabetes].sample(frac=0.5, random_state=42).index
    df_real_encoded.loc[diabetes_indices, 'Gender_Male'] = np.nan  # Introduce missing values for Gender

    # Example: Introduce missing values in 'Diabetes' for Males (using one-hot encoded columns)
    condition_male = df_real_encoded['Gender_Male'] == 1
    male_indices = df_real_encoded[condition_male].sample(frac=0.5, random_state=42).index
    df_real_encoded.loc[male_indices, 'Diabetes_Yes'] = np.nan  # Introduce missing values for Diabetes

    return df_real_encoded

# Step 3: Apply the function to introduce MAR missing values
df_real_mar = introduce_mar(df_real_encoded)

# Step 4: Check the missing data summary after introducing MAR
print(df_real_mar.isnull().sum())

#EDA after introducting missing values
# Path to your output Excel file
output_path = r"C:\Users\khaiy\OneDrive\Documents\Capstone Project 2\Summ_Stats.xlsx"

# Summary statistics for numerical variables including NaN values
numerical_summary_mar = df_real_mar.describe(include='all')

# Print numerical summary including NaN values
print("Numerical Summary_mar (including NaN values):")
print(numerical_summary_mar)

# Save the numerical summary to the Excel file in a new sheet
with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
    # Save numerical summary to a new sheet named 'New_SummaryStats'
    numerical_summary_mar.to_excel(writer, sheet_name='New_SummaryStats')

print(f"Numerical summary statistics saved to a new sheet in: {output_path}")



# Checking missing values by column
print("\nMissing Values Summary:\n", df_real_mar.isnull().sum())

# Visualizing the missing values with a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_real_mar.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

#Distribution of each numerical column
# Histograms for numerical columns
# Plot Distribution and Histogram for the numerical columns
# Select columns starting with 'Exercise Hours' and add to selected columns list
exercise_hours_columns = [col for col in df_real_mar.columns if col.startswith('Exercise Hours')]
selected_columns_missing = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Stress Level', 'Blood Sugar'] + exercise_hours_columns
num_plots = len(selected_columns_missing)

# Define the number of rows and columns for the subplot grid
n_cols = 3  # Number of columns per row
n_rows = (num_plots // n_cols) + (num_plots % n_cols > 0)  # Calculate number of rows needed

# Set up the subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
fig.suptitle("Distributions of Numerical Columns After Preprocessing", fontsize=16, y=1.02)

# Flatten axes array if necessary (for easy iteration)
axes = axes.flatten()

# Set the color for differentiating original vs after introducing missing values
color = 'violet'

# Plot each numerical column's distribution
for i, column in enumerate(selected_columns_missing):
    sns.histplot(df_real_mar[column], kde=True, bins=30, color=color, ax=axes[i])
    axes[i].set_title(f"Distribution of {column}")
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Frequency")

# Remove any empty subplots if there are extra grid spaces
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Step 5: Count plots for each categorical variable (one-hot encoded)

# Identify categorical columns
categorical_columns_missing = df_real_mar.select_dtypes(include=['uint8']).columns

# Determine the number of categorical variables
num_plots = len(categorical_columns_missing)

# Define the number of rows and columns for the subplot grid
n_cols = 3  # Number of columns per row
n_rows = (num_plots // n_cols) + (num_plots % n_cols > 0)  # Calculate number of rows needed

# Set up the subplot grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
fig.suptitle("Count Plots for Categorical Variables", fontsize=16, y=1.02)

# Flatten axes array if necessary (for easy iteration)
axes = axes.flatten()

# Plot each categorical column's count plot
for i, column in enumerate(categorical_columns_missing):
    sns.countplot(x=column, data=df_real_mar, ax=axes[i])
    axes[i].set_title(f"Count of {column}")
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Count")

# Remove any empty subplots if there are extra grid spaces
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Manually select specific columns for correlation
selected_columns_corr = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Exercise Hours_1', 'Stress Level', 'Blood Sugar','Heart Disease']   

# Correlation matrix
corr_matrix_mar = df_real_mar[selected_columns_corr].corr()
# Heatmap for correlation
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix_mar, annot=True, cmap='YlGnBu')
plt.show()

# Manually select numerical columns
selected_columns_mar = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Exercise Hours_1', 'Stress Level', 'Blood Sugar', 'Heart Disease']   

#Perform Three Imputation Methods (Prioritize USE)
# Assume df_real_mar is the dataset with missing values introduced from the previous steps

def mean_imputation(df):
    imputer = SimpleImputer(strategy='mean')
    df_mean_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['float64', 'int64'])), columns=df.select_dtypes(include=['float64', 'int64']).columns)
    
    # Convert numerical columns to integers
    df_mean_imputed = df_mean_imputed.astype(int)

    # Handle categorical columns (convert to 0 or 1)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df_mean_imputed[col] = df[col].fillna(0).map(lambda x: 1 if x != 0 else 0)

    return df_mean_imputed


# Step 2: KNN Imputation
def knn_imputation(df, n_neighbors=5):
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df.select_dtypes(include=['float64', 'int64'])), columns=df.select_dtypes(include=['float64', 'int64']).columns)
    
    # Convert numerical columns to integers
    df_knn_imputed = df_knn_imputed.astype(int)

    # Handle categorical columns (convert to 0 or 1)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df_knn_imputed[col] = df[col].fillna(0).map(lambda x: 1 if x != 0 else 0)

    return df_knn_imputed

#Regression Imputation
def hgb_imputation(df, target_column):
    # Define the target variable and features (excluding the target column from features)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check for non-NaN rows in the target column
    not_nan_indices = y.notnull()  # Indices where the target column is not NaN
    if not any(not_nan_indices):  # If there are no non-NaN values, skip imputation
        print(f"No valid samples to fit for {target_column}. Skipping imputation.")
        return df

    # Create and fit the HGB model only on the valid data
    model = HistGradientBoostingRegressor()
    model.fit(X[not_nan_indices], y[not_nan_indices])
    
    # Predict missing values for rows where the target column is NaN
    nan_indices = y.isnull()  # Indices where the target column is NaN
    if any(nan_indices):  # Only predict if there are missing values
        df.loc[nan_indices, target_column] = np.round(model.predict(X[nan_indices])).astype(int)


    # Convert categorical columns to 0 or 1
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].fillna(0).map(lambda x: 1 if x != 0 else 0)

    return df

# Apply imputations and save results to different DataFrames
# Make a copy of the dataset to avoid altering the original dataset
df_real_mean = df_real_mar.copy()
df_real_knn = df_real_mar.copy()
df_real_hgb = df_real_mar.copy()

# Step 4: Apply Mean Imputation
df_real_mean = mean_imputation(df_real_mean)

# Step 5: Apply KNN Imputation
df_real_knn = knn_imputation(df_real_knn, n_neighbors=5)

# Apply HGB imputation for selected columns
# Step 6: Apply regression imputation for numerical columns
df_real_hgb = hgb_imputation(df_real_hgb, target_column='Cholesterol')
df_real_hgb = hgb_imputation(df_real_hgb, target_column='Blood Pressure')
df_real_hgb = hgb_imputation(df_real_hgb, target_column='Heart Rate')
df_real_hgb = hgb_imputation(df_real_hgb, target_column='Blood Sugar')

# Apply regression imputation for one-hot encoded categorical columns
df_real_hgb = hgb_imputation(df_real_hgb, target_column='Age')
df_real_hgb = hgb_imputation(df_real_hgb, target_column='Gender_Male')
df_real_hgb = hgb_imputation(df_real_hgb, target_column='Diabetes_Yes')


# Save each imputed dataset to a CSV file
df_real_mean.to_csv('df_real_mean_imputed.csv', index=False)
df_real_knn.to_csv('df_real_knn_imputed.csv', index=False)
df_real_hgb.to_csv('df_real_regression_imputed.csv', index=False)

# Print missing values count for each dataset after imputation
print("Mean Imputation Missing Values:\n", df_real_mean.isnull().sum())
print("KNN Imputation Missing Values:\n", df_real_knn.isnull().sum())
print("Regression Imputation Missing Values:\n", df_real_hgb.isnull().sum())

# Function to perform Agglomerative Clustering
def agglomerative_clustering(df, n_clusters=3, method='complete'):
    # Scaling the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)  # Assuming df is fully numeric and imputed

    # Apply Agglomerative Clustering
    agglo_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=method)
    cluster_labels = agglo_cluster.fit_predict(df_scaled)

    return cluster_labels  # Returning labels for further use or visualization

#Dendrogram Visualization
# Function to plot dendrograms for each dataset
def plot_dendrograms(data_list, titles):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Create a row of 3 plots

    for i, (data, title) in enumerate(zip(data_list, titles)):
        # Perform hierarchical clustering (linkage) for dendrogram
        linked = linkage(data, method='complete')  # You can choose other methods like 'average', 'complete', etc.

        # Plot the dendrogram
        dendrogram(linked, ax=axes[i], truncate_mode='lastp', p=20, leaf_rotation=45, leaf_font_size=10)
        axes[i].set_title(title)
        axes[i].set_xlabel('Sample index or (cluster size)')
        axes[i].set_ylabel('Distance')

    plt.tight_layout()
    plt.show()

# Create a list of data and titles for each plot
data_list = [df_real_mean, df_real_knn, df_real_hgb]
titles = ['Dendrogram for Mean Imputed Data', 'Dendrogram for KNN Imputed Data', 'Dendrogram for Regression Imputed Data']

# Call the function to create a combined dendrogram plot
plot_dendrograms(data_list, titles)


# Function to evaluate clustering performance
def evaluate_clustering(df, cluster_labels):
    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(df, cluster_labels)
    # Calculate Davies-Bouldin Index
    davies_bouldin = davies_bouldin_score(df, cluster_labels)

    return silhouette_avg, davies_bouldin


# Function to calculate Dunn Index
def dunn_index(df_scaled, cluster_labels):
    # Intra-cluster distances (compactness)
    intra_distances = []
    for i in set(cluster_labels):
        cluster_points = df_scaled[cluster_labels == i]
        pairwise_distances = np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2)
        max_intra_distance = np.max(pairwise_distances)  # Max distance within the cluster
        intra_distances.append(max_intra_distance)
    max_intra = np.max(intra_distances)  # Largest intra-cluster distance across all clusters

    # Inter-cluster distances (separation)
    inter_distances = []
    unique_clusters = list(set(cluster_labels))
    for i, cluster_i in enumerate(unique_clusters):
        for j, cluster_j in enumerate(unique_clusters):
            if i < j:  # Avoid duplicate calculations
                points_i = df_scaled[cluster_labels == cluster_i]
                points_j = df_scaled[cluster_labels == cluster_j]
                inter_dist = np.min(np.linalg.norm(points_i[:, np.newaxis] - points_j, axis=2))
                inter_distances.append(inter_dist)
    min_inter = np.min(inter_distances)  # Smallest inter-cluster distance

    # Calculate Dunn Index
    dunn_index_value = min_inter / max_intra
    return dunn_index_value

# Initialize DataFrame to store clustering evaluation results
results_df = pd.DataFrame(columns=['Imputation Method', 'Silhouette Score', 'Davies-Bouldin Index', 'Dunn Index'])

# Perform clustering and evaluations for each imputed dataset
# 1. Mean Imputation
mean_cluster_labels = agglomerative_clustering(df_real_mean)
mean_silhouette, mean_db_index = evaluate_clustering(df_real_mean, mean_cluster_labels)
scaler = StandardScaler()
df_real_mean_scaled = scaler.fit_transform(df_real_mean)  # Standardize for Dunn Index calculation
mean_dunn_index = dunn_index(df_real_mean_scaled, mean_cluster_labels)
df_real_mean['Cluster'] = mean_cluster_labels
results_df = results_df.append({
    'Imputation Method': 'Mean Imputation',
    'Silhouette Score': mean_silhouette,
    'Davies-Bouldin Index': mean_db_index,
    'Dunn Index': mean_dunn_index
}, ignore_index=True)

# 2. KNN Imputation
knn_cluster_labels = agglomerative_clustering(df_real_knn)
knn_silhouette, knn_db_index = evaluate_clustering(df_real_knn, knn_cluster_labels)
df_real_knn_scaled = scaler.fit_transform(df_real_knn)  # Standardize for Dunn Index calculation
knn_dunn_index = dunn_index(df_real_knn_scaled, knn_cluster_labels)
df_real_knn['Cluster'] = knn_cluster_labels
results_df = results_df.append({
    'Imputation Method': 'KNN Imputation',
    'Silhouette Score': knn_silhouette,
    'Davies-Bouldin Index': knn_db_index,
    'Dunn Index': knn_dunn_index
}, ignore_index=True)

# 3. Regression Imputation
regression_cluster_labels = agglomerative_clustering(df_real_hgb)
regression_silhouette, regression_db_index = evaluate_clustering(df_real_hgb, regression_cluster_labels)
df_real_hgb_scaled = scaler.fit_transform(df_real_hgb)  # Standardize for Dunn Index calculation
regression_dunn_index = dunn_index(df_real_hgb_scaled, regression_cluster_labels)
df_real_hgb['Cluster'] = regression_cluster_labels
results_df = results_df.append({
    'Imputation Method': 'Regression Imputation',
    'Silhouette Score': regression_silhouette,
    'Davies-Bouldin Index': regression_db_index,
    'Dunn Index': regression_dunn_index
}, ignore_index=True)

# Display the results DataFrame
print(results_df)

# Extract metrics from results DataFrame for plotting
methods = results_df['Imputation Method']
silhouette_scores = results_df['Silhouette Score']
davies_bouldin_indices = results_df['Davies-Bouldin Index']
dunn_indices = results_df['Dunn Index']

# Plotting
x = np.arange(len(methods))  # label locations
width = 0.25  # bar width

fig, ax = plt.subplots(figsize=(12, 6))

# Create bar plots for Silhouette Score, Davies-Bouldin Index, and Dunn Index
bars1 = ax.bar(x - width, silhouette_scores, width, label='Silhouette Score', color='b')
bars2 = ax.bar(x, davies_bouldin_indices, width, label='Davies-Bouldin Index', color='r')
bars3 = ax.bar(x + width, dunn_indices, width, label='Dunn Index', color='g')

# Labels and title
ax.set_ylabel('Scores')
ax.set_title('Clustering Evaluation Metrics by Imputation Method')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

# Add value annotations on the bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.tight_layout()
plt.show()


