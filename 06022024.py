# --- Imports ---
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from prophet import Prophet
import matplotlib.pyplot as plt

# --- Download NLTK resources ---
nltk.download('punkt')
nltk.download('stopwords')

# --- Example DataFrame (Replace with your real data) ---
data = {
    'time': pd.date_range(start='2015-01-01', periods=100, freq='D'),
    'dialogue': [f"Example dialogue {i} with some random text" for i in range(100)]
}
df = pd.DataFrame(data)

# --- Time Feature Enrichment Function ---
def enrich_time_features(df, time_column='time'):
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])

    df['hour'] = df[time_column].dt.hour
    df['minute'] = df[time_column].dt.minute
    df['second'] = df[time_column].dt.second
    df['day'] = df[time_column].dt.day
    df['weekday'] = df[time_column].dt.weekday
    df['weekday_name'] = df[time_column].dt.day_name()
    df['date'] = df[time_column].dt.date
    df['year'] = df[time_column].dt.year
    df['month'] = df[time_column].dt.month
    df['month_name'] = df[time_column].dt.month_name()
    df['week'] = df[time_column].dt.isocalendar().week
    df['day_of_year'] = df[time_column].dt.dayofyear
    df['quarter'] = df[time_column].dt.quarter
    df['is_month_start'] = df[time_column].dt.is_month_start
    df['is_month_end'] = df[time_column].dt.is_month_end
    df['year_week'] = df[time_column].dt.strftime('%Y-%U')
    df['elapsed_seconds'] = (df[time_column] - df[time_column].min()).dt.total_seconds()
    
    return df

# --- Natural Language Processing Function ---
def preprocess_text(text):
    """
    Basic text cleaning and tokenization.
    """
    # Lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# --- Apply Time Feature Extraction ---
df = enrich_time_features(df)

# --- Apply Text Preprocessing ---
df['tokens'] = df['dialogue'].apply(preprocess_text)

# --- Build a Basic Metric: Dialogue Length (number of tokens) ---
df['dialogue_length'] = df['tokens'].apply(len)

# --- Prepare Data for Prophet ---
# Prophet expects columns ['ds', 'y']
prophet_df = df[['time', 'dialogue_length']].rename(columns={'time': 'ds', 'dialogue_length': 'y'})

# --- Prophet Forecasting ---
model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
model.fit(prophet_df)

# Predict into the future
future = model.make_future_dataframe(periods=365)  # Predict 1 more year daily
forecast = model.predict(future)

# --- Plot Results ---
fig1 = model.plot(forecast)
plt.title("Forecast of Dialogue Complexity Over Time (Token Length)")
plt.xlabel("Date")
plt.ylabel("Dialogue Length (Number of Words)")
plt.show()

# --- Show top 5 records for inspection ---
print(df[['time', 'dialogue', 'tokens', 'dialogue_length']].head())
