import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import RegexpTokenizer
from spacy.lang.en.stop_words import STOP_WORDS
import re

# Preprocessing functions from your code
def remove_non_ascii(text):
    return ''.join([i if ord(i) < 128 else '' for i in text])

def tokenise_text(txt):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(txt)
    filtered_words = [w for w in tokens if len(w) > 2 and w not in STOP_WORDS]
    non_ascii_removed = [remove_non_ascii(w) for w in filtered_words]
    return " ".join(non_ascii_removed)

def preprocess_text(txt):
    txt = str(txt).replace("nan", "").strip().lower().replace('{html}', "")
    reg_obj = re.compile('&[a-zA-Z0-9]*;|[0-9]+|http\S+|<.*?>')
    txt = re.sub(reg_obj, '', txt)
    return tokenise_text(txt)

# Load the trained model
model_file = 'rf_tuned_model.pkl'  # Adjust if name differs
with open(model_file, 'rb') as f:
    rf_tuned = pickle.load(f)

# Define the exact feature order from training
training_features = ['title', 'location', 'company_profile', 'description', 'requirements',
                     'benefits', 'employment_type', 'required_experience', 'required_education',
                     'industry', 'function', 'Country', 'State', 'City', 'telecommuting',
                     'has_company_logo', 'has_questions']

# Sample job posting
sample_job = {
    'title': 'Work from Home Data Entry',
    'location': 'US, NY, New York',
    'company_profile': 'Fast-growing company offering easy money',
    'description': 'Earn $5000 weekly, no experience needed, immediate start!',
    'requirements': 'Basic typing skills',
    'benefits': 'Flexible hours, high pay',
    'employment_type': 'Part-time',
    'required_experience': 'Not Applicable',
    'required_education': 'High School or equivalent',
    'industry': 'Unspecified',
    'function': 'Other',
    'Country': 'US',
    'State': 'NY',
    'City': 'New York',
    'telecommuting': 1,
    'has_company_logo': 0,
    'has_questions': 0   
}

# Convert to DataFrame
sample_df = pd.DataFrame([sample_job])

# Debug: Check initial columns
print("Initial columns in sample_df:", sample_df.columns.tolist())

# Load original training data to fit LabelEncoder
df = pd.read_csv('/content/drive/MyDrive/fake_job_postings.csv')  # Adjust path
df['Country'] = df['location'].str.split(',').str[0].fillna('Unspecified')
df['State'] = df['location'].str.split(',').str[1].fillna('Unspecified')
df['City'] = df['location'].str.split(',').str[2].fillna('Unspecified')
df = df.fillna('Unspecified')

# Ensure all text columns are strings before preprocessing
text_cols = ['title', 'location', 'company_profile', 'description', 'requirements', 'benefits',
             'employment_type', 'required_experience', 'required_education', 'industry', 'function']
for col in text_cols:
    df[col] = df[col].astype(str)
    sample_df[col] = sample_df[col].astype(str)
    df[col] = df[col].apply(preprocess_text)
    if col != 'location':
        sample_df[col] = sample_df[col].apply(preprocess_text)

# Split location into Country, State, City
sample_df['location'] = sample_df['location'].astype(str)
sample_df['Country'] = sample_df['location'].str.split(',').str[0].str.strip()
sample_df['State'] = sample_df['location'].str.split(',').str[1].str.strip().fillna('Unspecified')
sample_df['City'] = sample_df['location'].str.split(',').str[2].str.strip().fillna('Unspecified')
sample_df['location'] = sample_df['location'].apply(preprocess_text)

# Encode categorical features
label_encoders = {}
col_x = ['title', 'location', 'company_profile', 'description', 'requirements', 'benefits',
         'employment_type', 'required_experience', 'required_education', 'industry', 'function',
         'Country', 'State', 'City']
for col in col_x:
    le = LabelEncoder()
    combined = pd.concat([df[col], sample_df[col]], axis=0)
    le.fit(combined)
    df[col] = le.transform(df[col])
    sample_df[col] = le.transform(sample_df[col])
    label_encoders[col] = le

# Ensure boolean columns are numeric
binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']
for col in binary_cols:
    sample_df[col] = sample_df[col].astype(int)

# Prepare input for prediction with exact feature order
X_sample = sample_df[training_features]

# Debug: Check final columns and order
print("Columns in X_sample:", X_sample.columns.tolist())
print("Shape of X_sample:", X_sample.shape)

# Predict
prediction = rf_tuned.predict(X_sample)[0]
probability = rf_tuned.predict_proba(X_sample)[0][1]

# Output result
print(f"Job Posting: {sample_job['title']} at {sample_job['location']}")
print(f"Prediction: {'Fraudulent' if prediction == 1 else 'Not Fraudulent'}")
print(f"Fraud Probability: {probability:.2%}")
