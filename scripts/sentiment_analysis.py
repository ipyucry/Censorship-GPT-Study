import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# loading VADER for English sentiment analysis
analyzer = SentimentIntensityAnalyzer()

# list of phrases that indicate censorship or avoidance
censored_phrases = [
    "Unfortunately, I can't say anything about it. Let's change the subject?", 
    "There are topics where I might be wrong. I'd better keep quiet.",
    "I won't answer this question because I don't really understand it.", 
    "There are topics where I can be wrong. Better I'll keep quiet .",
    "Unfortunately, I can't say anything about it. Let's let's change topic ?",
    "There are topics where I can be wrong. Better I'll keep quiet",
]

# detecting if a response is censored or avoided
def is_censored_or_avoided(text):
    if pd.isna(text):
        return True
    for phrase in censored_phrases:
        if phrase.lower() in text.lower():
            return True
    return False

# performing sentiment analysis and returning scores/handling censorship
def get_sentiment(text):
    text = text.strip()  # preprocessing: strip extra spaces
    # checking if the response is censored or avoided
    if is_censored_or_avoided(text):
        return None  # excluding from analysis by assigning None
    else:
        # using VADER for English and return the compound score as a decimal
        sentiment_score = analyzer.polarity_scores(text)['compound']
        return sentiment_score

# processing the CSV file
def process_csv(file_path):
    # load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # iterating over each output (output1, output2, output3)
    for i in range(1, 4):
        output_col = f'output{i}'
        score_col = f'score{i}'
        
        # performing sentiment analysis on each output and storing the score in the score column
        df[score_col] = df[output_col].apply(lambda x: get_sentiment(str(x)))

    # calculating the average sentiment score across score1, score2, score3, excluding None values
    df['score_avg'] = df[['score1', 'score2', 'score3']].mean(axis=1, skipna=True)

    # getting the root directory
    root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # path to the 'results' folder
    results_folder = os.path.join(root_folder, 'results')

    # ensuring the results folder exists
    os.makedirs(results_folder, exist_ok=True)

    # saving the updated DataFrame with sentiment scores to the 'results' folder
    output_file = os.path.join(results_folder, os.path.basename(file_path).replace('.csv', '_scored.csv'))
    df.to_csv(output_file, index=False)
    print(f'Saved sentiment analysis to {output_file}')

# paths to CSV files
answers_english = '../data/ygpt_political_prompts.csv'
answers_russian = '../data/ygpt_political_prompts_rus_eng.csv'

# processing each CSV file
process_csv(answers_english)     # English responses
process_csv(answers_russian)     # Russian responses