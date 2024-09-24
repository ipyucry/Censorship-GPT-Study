import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# loading the Sentence-BERT model for sentence embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# calculating cosine similarity between two sentences
def cosine_similarity(text1, text2):
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    return cosine_sim

# computing consistency score
def compute_consistency(output1, output2, output3):
    sim1_2 = cosine_similarity(output1, output2)
    sim2_3 = cosine_similarity(output2, output3)
    sim1_3 = cosine_similarity(output1, output3)
    
    # averaging the similarities
    avg_similarity = (sim1_2 + sim2_3 + sim1_3) / 3

    return avg_similarity

# processing the CSV file and computing consistency
def process_csv_for_consistency(file_name, data_folder, results_folder):
    # loading the CSV file into a DataFrame
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(file_path)

    # creating a column to store consistency score
    df['consistency'] = df.apply(lambda row: compute_consistency(str(row['output1']), 
                                                                 str(row['output2']), 
                                                                 str(row['output3'])), axis=1)

    # calculating the average consistency score for the entire dataset
    average_consistency = df['consistency'].mean()

    # creating a new column called 'average_consistency' and setting its value to the average consistency
    df['average_consistency'] = average_consistency

    # making sure only the first row has the average consistency value, the rest are NaN
    df['average_consistency'] = df['average_consistency'].where(df.index == 0)
    
    # keeping only the relevant columns (id, output1, output2, output3, consistency, average_consistency)
    result_df = df[['id', 'output1', 'output2', 'output3', 'consistency', 'average_consistency']]

    # sorting the DataFrame by consistency score (descending)
    result_df = result_df.sort_values(by='consistency', ascending=False)

    # generating the output file path
    output_file = os.path.join(results_folder, file_name.replace('.csv', '_consistency.csv'))

    # saving the updated DataFrame to the results folder
    result_df.to_csv(output_file, index=False)
    print(f'Saved consistency analysis to {output_file}')

# main function to run consistency analysis for all files
def run_consistency_analysis():
    # getting the root directory of the project
    root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # defining paths for the data and results folders
    data_folder = os.path.join(root_folder, 'data')
    results_folder = os.path.join(root_folder, 'results')

    # ensure the results folder exists
    os.makedirs(results_folder, exist_ok=True)

    # list of files to process
    file_names = ['ygpt_political_prompts.csv', 'ygpt_political_prompts_rus_eng.csv']

    # processing each CSV file and computing consistency
    for file_name in file_names:
        process_csv_for_consistency(file_name, data_folder, results_folder)

# run the consistency analysis
run_consistency_analysis()