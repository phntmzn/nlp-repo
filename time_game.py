import numpy as np
import pandas as pd
import random
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd

# Example payoff matrices for each sender type
payoff_matrix_type1 = np.array([
    [3, 1, 0],  # Response 1, 2, 3 for Message A
    [2, 3, 1]   # Response 1, 2, 3 for Message B
])

payoff_matrix_type2 = np.array([
    [1, 0, 3],  # Response 1, 2, 3 for Message A
    [0, 3, 2]   # Response 1, 2, 3 for Message B
])


# Simulate a time series of sender types (Type 1 and Type 2)
time_series_data = pd.Series(np.random.choice([1, 2], size=100))

# Analyze how often each type sends each message over time
message_history = pd.DataFrame({
    'Type': time_series_data,
    'Message_A': np.random.binomial(1, 0.5, size=100),  # Randomly simulate Message A being sent
    'Message_B': np.random.binomial(1, 0.5, size=100)   # Randomly simulate Message B being sent
})

# Calculate moving average or other time series metrics
message_history['Message_A_MA'] = message_history['Message_A'].rolling(window=5).mean()
message_history['Message_B_MA'] = message_history['Message_B'].rolling(window=5).mean()


# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Assume messages are actual text (for simplicity using short phrases here)
messages = ["The market is up", "The market is down", "Hold", "Sell"]

# Analyze sentiment of each message
sentiments = [sia.polarity_scores(message)['compound'] for message in messages]

# Mapping sentiment to game strategy (example logic)
strategies = ["A" if sentiment > 0 else "B" for sentiment in sentiments]

for t in range(len(time_series_data)):
    sender_type = time_series_data.iloc[t]
    
    # Determine the message to send based on time series (e.g., historical strategy)
    if sender_type == 1:
        chosen_message = 'A' if message_history['Message_A_MA'].iloc[t] > 0.5 else 'B'
    else:
        chosen_message = 'B' if message_history['Message_B_MA'].iloc[t] > 0.5 else 'A'
    
    # Determine response based on payoffs and sentiment (example simple logic)
    if chosen_message == 'A':
        response = np.argmax(payoff_matrix_type1[0]) + 1
    else:
        response = np.argmax(payoff_matrix_type2[1]) + 1
    
    print(f"Round {t+1}: Sender Type {sender_type}, Message {chosen_message}, Receiver Response {response}")

# Define a peer pressure influence parameter (0 = no influence, 1 = full influence)
peer_pressure_influence = 0.7
mitigation_factor = 0.5  # Reduces the peer pressure influence

# Example sender type and message history
sender_types = np.random.choice([1, 2], size=100)
message_history = np.random.choice(['A', 'B'], size=100)

def decide_message(sender_type, history, peer_pressure_influence, mitigation_factor):
    # Calculate the influence of peer pressure
    influence = peer_pressure_influence * (1 - mitigation_factor)
    
    if sender_type == 1:
        # Peer pressure suggests following the majority choice
        if np.sum(history == 'A') > np.sum(history == 'B'):
            biased_choice = 'A'
        else:
            biased_choice = 'B'
    else:
        if np.sum(history == 'B') > np.sum(history == 'A'):
            biased_choice = 'B'
        else:
            biased_choice = 'A'
    
    # Mitigation: Introduce a probability of independent choice
    independent_choice = random.choice(['A', 'B'])
    return biased_choice if random.random() < influence else independent_choice

# Simulate the game with mitigation of peer pressure
for t in range(1, len(sender_types)):
    sender_type = sender_types[t]
    chosen_message = decide_message(sender_type, message_history[:t], peer_pressure_influence, mitigation_factor)
    message_history[t] = chosen_message

    print(f"Round {t+1}: Sender Type {sender_type}, Chosen Message {chosen_message}")

def decide_response(sender_type, message, peer_pressure_influence, mitigation_factor, response_history):
    # Calculate the influence of peer pressure
    influence = peer_pressure_influence * (1 - mitigation_factor)
    
    if message == 'A':
        biased_response = np.argmax(payoff_matrix_type1[0]) + 1
    else:
        biased_response = np.argmax(payoff_matrix_type2[1]) + 1
    
    # Mitigation: Introduce a probability of independent response
    independent_response = random.choice([1, 2, 3])
    return biased_response if random.random() < influence else independent_response

# Simulate the receiver's decision with peer pressure mitigation
response_history = []

for t in range(1, len(sender_types)):
    sender_type = sender_types[t]
    chosen_message = message_history[t]
    response = decide_response(sender_type, chosen_message, peer_pressure_influence, mitigation_factor, response_history)
    response_history.append(response)
    
    print(f"Round {t+1}: Sender Type {sender_type}, Message {chosen_message}, Receiver Response {response}")


# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Example payoff matrices for each sender type
payoff_matrix_type1 = np.array([[3, 1, 0], [2, 3, 1]])
payoff_matrix_type2 = np.array([[1, 0, 3], [0, 3, 2]])

# Time series: Initialize sender types and message history
num_rounds = 100
sender_types = np.random.choice([1, 2], size=num_rounds)
message_history = np.random.choice(['A', 'B'], size=num_rounds)

# Peer pressure parameters
peer_pressure_influence = 0.7
mitigation_factor = 0.5  # Reduces the peer pressure influence

# Time series data
time_series_data = pd.DataFrame({
    'Round': np.arange(num_rounds),
    'Sender_Type': sender_types,
    'Message': message_history,
    'Sentiment': np.zeros(num_rounds),
    'Response': np.zeros(num_rounds, dtype=int)
})

# NLP analysis and decision-making
def analyze_message(message):
    sentiment = sia.polarity_scores(message)['compound']
    return sentiment

def decide_message(sender_type, history, peer_pressure_influence, mitigation_factor, sentiment):
    influence = peer_pressure_influence * (1 - mitigation_factor)
    
    if sentiment > 0:
        independent_choice = 'A'
    else:
        independent_choice = 'B'
    
    if sender_type == 1:
        biased_choice = 'A' if np.sum(history == 'A') > np.sum(history == 'B') else 'B'
    else:
        biased_choice = 'B' if np.sum(history == 'B') > np.sum(history == 'A') else 'A'
    
    return biased_choice if random.random() < influence else independent_choice

def decide_response(sender_type, message, peer_pressure_influence, mitigation_factor, sentiment, response_history):
    influence = peer_pressure_influence * (1 - mitigation_factor)
    
    if message == 'A':
        biased_response = np.argmax(payoff_matrix_type1[0]) + 1
    else:
        biased_response = np.argmax(payoff_matrix_type2[1]) + 1
    
    independent_response = random.choice([1, 2, 3])
    return biased_response if random.random() < influence else independent_response

# Simulate the game with NLP and time series tracking
for t in range(num_rounds):
    sender_type = time_series_data['Sender_Type'].iloc[t]
    current_message = time_series_data['Message'].iloc[t]
    
    # NLP Analysis
    sentiment = analyze_message(current_message)
    time_series_data.at[t, 'Sentiment'] = sentiment
    
    # Decide message considering peer pressure and NLP
    chosen_message = decide_message(sender_type, message_history[:t], peer_pressure_influence, mitigation_factor, sentiment)
    message_history[t] = chosen_message
    time_series_data.at[t, 'Message'] = chosen_message
    
    # Decide response considering peer pressure and NLP
    response = decide_response(sender_type, chosen_message, peer_pressure_influence, mitigation_factor, sentiment, time_series_data['Response'][:t])
    time_series_data.at[t, 'Response'] = response

    print(f"Round {t+1}: Sender Type {sender_type}, Chosen Message {chosen_message}, Sentiment {sentiment}, Receiver Response {response}")

# Analyze and visualize the time series data

# Plotting the frequency of each message over time
time_series_data['Message_A_Count'] = (time_series_data['Message'] == 'A').cumsum()
time_series_data['Message_B_Count'] = (time_series_data['Message'] == 'B').cumsum()

plt.plot(time_series_data['Round'], time_series_data['Message_A_Count'], label='Message A Count')
plt.plot(time_series_data['Round'], time_series_data['Message_B_Count'], label='Message B Count')
plt.xlabel('Round')
plt.ylabel('Cumulative Count')
plt.title('Message Frequency Over Time')
plt.legend()
plt.show()

