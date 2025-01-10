# # Project Reviews Aggregator

# %% [markdown]
# ## Import libraries

# %%
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# %% [markdown]
# ## Data cleaning

# %% [markdown]
# ### Load Kaggle data

# %%
# Loading data from kaggle

import kagglehub

# Download latest version
path_kaggle = kagglehub.dataset_download("datafiniti/consumer-reviews-of-amazon-products")

print("Path to dataset files:", path_kaggle)

# %%
# List all CSV files in the directory
csv_files = [file for file in os.listdir(path_kaggle) if file.endswith('.csv')]

# Iterate through each CSV file and print its headers
for file in csv_files:
    file_path = os.path.join(path_kaggle, file)
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Headers for {file}:")
    for header in df.columns:
        print(header)
    print("\n" + "-" * 50 + "\n")  # Separator for better readability


# %%
######## Create Kaggle dataframe with the 3 review csv(commented to avoid running mulitple times) ########

"""
# List all CSV files in the directory
csv_files = [file for file in os.listdir(path_kaggle) if file.endswith('.csv')]

# Combine all CSV files into a single DataFrame
dataframes = []
for file in csv_files:
    file_path = os.path.join(path_kaggle, file)
    df = pd.read_csv(file_path, low_memory=False)  # Prevent dtype warnings
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame, aligning columns
kag_comb = pd.concat(dataframes, ignore_index=True, sort=True)

# Define the project directory
project_dir = os.getcwd()  # Gets the current working directory

# Path to the "data" folder in the project directory
data_dir = os.path.join(project_dir, "data")

# Ensure the "data" folder exists
os.makedirs(data_dir, exist_ok=True)

# Save the combined DataFrame to a new CSV file in the "data" folder
kag_comb_path = os.path.join(data_dir, "kag_comb.csv")
kag_comb.to_csv(kag_comb_path, index=False)

print(f"Combined CSV saved to {kag_comb_path}")
"""


######## Load generated CSV ########

kag_comb = pd.read_csv("data/kag_comb.csv")


# %% [markdown]
# ### Clean Kaggle data

# %%
#Check df heads

# List of columns to drop
columns_to_drop = [
    'dateAdded',
    'dateUpdated',
    'reviews.didPurchase',
    'reviews.userCity',
    'reviews.userProvince',
    'asins',
    'imageURLs',
    'manufacturerNumber',
    'primaryCategories',
    'sourceURLs',
    'reviews.sourceURLs',
    'keys',
    'reviews.date',
    'reviews.dateAdded',
    'reviews.dateSeen',
    'reviews.numHelpful'
]

# Drop the columns from the dataframe
kag_comb = kag_comb.drop(columns=columns_to_drop)


# Get a list of all column headers
headers = kag_comb.columns.tolist()

# Display the headers
for header in headers:
    print(header)


# %%
kag_comb.head()

# %%
# Check for duplicate rows based on 'reviews.text', 'reviews.username', 'reviews.id' columns
duplicates = kag_comb.duplicated(subset=['reviews.text', 'reviews.username', 'reviews.id'])

# Count the number of duplicate rows
num_duplicates = duplicates.sum()
print(f"Number of duplicate rows based on 'reviews.text', 'reviews.username', 'reviews.id': {num_duplicates}")

# If you want to display the duplicate rows
if num_duplicates > 0:
    duplicate_rows = kag_comb[duplicates]
    print("Duplicate rows based on 'reviews.text', 'reviews.username', 'reviews.id':")
    print(duplicate_rows)


# %%
# Create a new dataframe with duplicates removed based on 'reviews.text', 'reviews.username' and 'id'
kag_comb_clean = kag_comb.drop_duplicates(subset=['reviews.text', 'reviews.username', 'reviews.id'])

# Remove rows with NaN in 'reviews.rating'
kag_comb_clean = kag_comb_clean.dropna(subset=['reviews.rating'])

# Remove rows with NaN in 'reviews.title'
kag_comb_clean = kag_comb_clean.dropna(subset=['reviews.title'])

# Verify the new dataframe
print(f"New dataframe shape: {kag_comb_clean.shape}")


# %%
# Check for NaN values in the 'reviews.ratingtext' column
nan_reviews_ratingtext = kag_comb_clean['reviews.rating'].isna()

# Count the number of NaN values
num_nan_reviews_ratingtext = nan_reviews_ratingtext.sum()
print(f"Number of NaN values in 'reviews.rating': {num_nan_reviews_ratingtext}")

# Display rows with NaN values in 'reviews.ratingtext'
if num_nan_reviews_ratingtext > 0:
    rows_with_nan = kag_comb_clean[nan_reviews_ratingtext]
    print("Rows with NaN in 'reviews.rating':")
    print(rows_with_nan)

# %% [markdown]
# ### Add missing data for classification training

# %%
# Fill NaN values in 'reviews.doRecommend' based on 'reviews.rating'
kag_comb_clean['reviews.doRecommend'] = kag_comb_clean['reviews.doRecommend'].fillna(
    kag_comb_clean['reviews.rating'] > 3
)

# Verify the changes
print(kag_comb_clean['reviews.doRecommend'].isna().sum())

# %%
# Check for NaN values in the 'reviews.doRecommend' column
nan_reviews_text = kag_comb_clean['reviews.doRecommend'].isna()

# Count the number of NaN values
num_nan_reviews_text = nan_reviews_text.sum()
print(f"Number of NaN values in 'reviews.doRecommend': {num_nan_reviews_text}")

# Display rows with NaN values in 'reviews.text'
if num_nan_reviews_text > 0:
    rows_with_nan = kag_comb_clean[nan_reviews_text]
    print("Rows with NaN in 'reviews.doRecommend':")
    print(rows_with_nan)

# %% [markdown]
# ## Preprocessing for sentiment training

# %% [markdown]
# ### Adding sentiment traning targets

# %%
def classify_sentiment(rating):
    if rating <= 2:
        return "NEG"
    elif rating == 3:
        return "NEU"
    else:
        return "POS"

kag_comb_clean['sentiment'] = kag_comb_clean['reviews.rating'].apply(classify_sentiment)

kag_comb_clean.head()

# %% [markdown]
# # Implementing PySentimiento for sentiment analysis

# %% [markdown]
# Selected as it is the most accurate model according to this benchmark:<br/>
# https://medium.com/@pavlo.fesenko/best-open-source-models-for-sentiment-analysis-part-2-neural-networks-9749fb5fff76<br/>
# Strategy is to generate sentiment in new columns through GPUs in Google colab and to go back to CPU in VScode for the rest of the project.

""""" PySentimiento Code
# %%
# def classify_sentiment_with_scores(text):
#     if not isinstance(text, str) or text.strip() == "":
#         return None  # Handle empty or non-string values gracefully
    
#     result = analyzer.predict(text)
#     return {
#         "label": result.output,  # Sentiment label
#         "probas": result.probas  # Probability scores
#     }

# kag_comb_clean['title sentiment details'] = kag_comb_clean['clean title'].apply(classify_sentiment_with_scores)
# kag_comb_clean['review sentiment details'] = kag_comb_clean['clean review'].apply(classify_sentiment_with_scores)


# def merge_probabilities(row):
#     if row['title sentiment details'] and row['review sentiment details']:
#         title_probas = row['title sentiment details']['probas']
#         review_probas = row['review sentiment details']['probas']
#         # Average the probabilities
#         combined_probas = {k: (title_probas[k] + review_probas[k]) / 2 for k in title_probas}
#         # Return the label with the highest combined probability
#         return max(combined_probas, key=combined_probas.get)
#     elif row['title sentiment details']:
#         return row['title sentiment details']['label']
#     elif row['review sentiment details']:
#         return row['review sentiment details']['label']
#     return None  # Handle empty rows

# kag_comb_clean['merged sentiment'] = kag_comb_clean.apply(merge_probabilities, axis=1)

"""
# %% [markdown]
# **<span style="color: #ff0000;"> PySentimiento is taking superlong to run even with GPU, moving to Vader in VS Code while Sentimiento runs on Colab.</span>**
# 
# Detailed results available in Main_PySentimiento_Review_evaluation
# 
# Generated with same logic implemented for Vader below and minimal text cleaning

# %%
# Importing PySentimiento generated CSV

kag_comb_clean_PySentimiento = pd.read_csv("data/kag_comb_clean_PySentimiento.csv")

# %%
kag_comb_clean_PySentimiento.head()

# %% [markdown]
# ### PySentimiento Evaluation

# %%
### PySentimiento Evaluation

# Compare ground-truth and predicted sentiment
kag_comb_clean_PySentimiento['PySent correct'] = kag_comb_clean_PySentimiento['sentiment'] == kag_comb_clean_PySentimiento['merged PySent sentiment']

pysent_accuracy = kag_comb_clean_PySentimiento['PySent correct'].mean()
print(f"PySent Accuracy: {pysent_accuracy:.2%}")



# Get ground truth and predictions
y_true = kag_comb_clean_PySentimiento['sentiment']
y_pred = kag_comb_clean_PySentimiento['merged PySent sentiment']

# Generate confusion matrix
labels = ['POS', 'NEU', 'NEG']  # Replace with the actual labels used in your dataset
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Print confusion matrix
print("Confusion Matrix:")

# Create a DataFrame for better visualization
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Plot confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# %%
from sklearn.metrics import classification_report, accuracy_score

# Generate classification report
report = classification_report(y_true, y_pred, labels=['POS', 'NEU', 'NEG'], target_names=['Positive', 'Neutral', 'Negative'])

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Print the classification report and accuracy
print("Classification Report:")
print(report)
print(f"Accuracy: {accuracy:.2%}")


# %% [markdown]
# # Implementing Vader

# %% [markdown]
# ### Functions for text cleaning

# %%
def clean_for_vader(text):
    """
    Cleans text for VADER by removing irrelevant artefacts while preserving sentiment-rich features.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Return the cleaned text without lowercasing or removing punctuation/emojis
    return text


# %%
kag_comb_clean['clean title for vader'] = kag_comb_clean['reviews.title'].apply(clean_for_vader)
kag_comb_clean['clean review for vader'] = kag_comb_clean['reviews.text'].apply(clean_for_vader)

# %%
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Function to classify sentiment using VADER
def classify_vader_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return None  # Handle empty or non-string values gracefully
    
    scores = sia.polarity_scores(text)  # Use the initialized sia object
    compound_score = scores['compound']  # Compound score is the overall sentiment score

    # Classify based on compound score
    if compound_score >= 0.05:
        return "POS"
    elif compound_score <= -0.05:
        return "NEG"
    else:
        return "NEU"
    
# Add sentiment score
def get_vader_scores(text):
    if not isinstance(text, str) or text.strip() == "":
        return None  # Handle empty or non-string values gracefully
    
    return sia.polarity_scores(text)  # Return the full score dictionary



# Apply VADER to the columns
kag_comb_clean['title sentiment - vader'] = kag_comb_clean['clean title for vader'].apply(classify_vader_sentiment)
kag_comb_clean['review sentiment- vader'] = kag_comb_clean['clean review for vader'].apply(classify_vader_sentiment)
kag_comb_clean['title sentiment score - vader'] = kag_comb_clean['clean title for vader'].apply(get_vader_scores)
kag_comb_clean['review sentiment score - vader'] = kag_comb_clean['clean review for vader'].apply(get_vader_scores)

# %%
def merge_vader_scores(row):
    title_score = row['title sentiment score - vader']['compound'] if row['title sentiment score - vader'] else 0
    review_score = row['review sentiment score - vader']['compound'] if row['review sentiment score - vader'] else 0

    # Calculate weighted sentiment score
    combined_score = (title_score*1.2 + review_score*1) / 2

    # Classify sentiment based on the combined score
    if combined_score >= 0.05:
        return "POS"
    elif combined_score <= -0.05:
        return "NEG"
    else:
        return "NEU"

# Apply the merging logic based on scores
kag_comb_clean['merged sentiment - vader'] = kag_comb_clean.apply(merge_vader_scores, axis=1)



# %%
kag_comb_clean.head(1)

# %%
# Compare ground-truth and predicted sentiment
kag_comb_clean['vader correct'] = kag_comb_clean['sentiment'] == kag_comb_clean['merged sentiment - vader']

# %%
vader_accuracy = kag_comb_clean['vader correct'].mean()
print(f"vader Accuracy: {vader_accuracy:.2%}")

# %%
kag_comb_clean.head(1)

# %% [markdown]
# ## Comparing results
# 
# It can be argued that ratings are not proper human validated labeling
# 
# Crossing the 2 models could provide outliers that are intesting to study for modle improvement
# 
# -> Will Get back here if I have time

# %% [markdown]
# # WINNER
# 
# Sentiment analysis using PySentimiento!
