import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# classifying prompts based on the ID column
def classify_prompts(df):
    def classify(row):
        if pd.isna(row['id']):
            return 'unknown'  # handle NaN IDs
        elif row['id'].startswith('US'):
            return 'U.S.'  # U.S.-related questions
        elif row['id'].startswith('RU'):
            return 'Russia'  # Russia-related questions
        else:
            return 'unknown'  # if country is unclear
    df['category'] = df.apply(classify, axis=1)
    return df

# calculating average sentiment scores for each category
def calculate_average_sentiment(df):
    avg_self_sentiment = df[df['category'] == 'Russia']['score_avg'].mean()
    avg_other_sentiment = df[df['category'] == 'U.S.']['score_avg'].mean()
    return avg_self_sentiment, avg_other_sentiment

# performing a t-test between self and other categories
def perform_t_test(df):
    self_scores = df[df['category'] == 'Russia']['score_avg'].dropna()
    other_scores = df[df['category'] == 'U.S.']['score_avg'].dropna()
    t_stat, p_value = stats.ttest_ind(self_scores, other_scores, equal_var=False)
    return t_stat, p_value

# plotting sentiment differences
def plot_sentiment_difference(df, avg_self, avg_other):
    plt.figure(figsize=(6, 4))

    # creating a boxplot to show sentiment distribution
    sns.boxplot(x='category', y='score_avg', data=df, palette='Set2')

    # adding the average sentiment as horizontal lines
    plt.axhline(avg_other, color='red', linestyle='--', label=f'U.S. Avg: {avg_other:.2f}')    
    plt.axhline(avg_self, color='blue', linestyle='--', label=f'Russia Avg: {avg_self:.2f}')

    # adjusting labels and title for compactness
    plt.title('Russian GPT', fontsize=10)
    plt.xlabel('Country', fontsize=9)
    plt.ylabel('Avg Sentiment Score', fontsize=9)

    # adjusting the legend font size
    plt.legend(fontsize=8, loc='best')

    # reducing tick label size
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # tightening the layout to reduce whitespace
    plt.tight_layout()

    # showing the plot
    plt.show()

# processing the CSV file and checking for bias
def check_for_bias(file_path):
    df = pd.read_csv(file_path)
    df = classify_prompts(df)

    # handling any missing or invalid sentiment scores (e.g., NaN values)
    df['score_avg'] = pd.to_numeric(df['score_avg'], errors='coerce')  # convert to numeric, handle non-numeric values
    df.dropna(subset=['score_avg'], inplace=True)  # drop rows where 'score_avg' is NaN

    # calculating the average sentiment for each category
    avg_self, avg_other = calculate_average_sentiment(df)
    print(f'Average sentiment for Russia: {avg_self}')
    print(f'Average sentiment for U.S.: {avg_other}')

    # performing a t-test to check if the difference in sentiment is statistically significant
    t_stat, p_value = perform_t_test(df)
    print(f'T-test statistic: {t_stat}, p-value: {p_value}')

    # interpreting the results
    if p_value < 0.05:
        print("The difference in sentiment between U.S. and Russia is statistically significant.")
    else:
        print("There is no statistically significant difference in sentiment between U.S. and Russia.")

    # plotting sentiment differences
    plot_sentiment_difference(df, avg_self, avg_other)

# CSV file path (select table for analysis)
csv_file_path = '../results/ygpt_political_prompts_rus_eng_scored.csv'

# run bias check
check_for_bias(csv_file_path)