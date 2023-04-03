import jsonlines
import pandas as pd
import numpy as np
import re
import nltk
import os
from nltk.tokenize import word_tokenize
from collections import Counter
from scipy.spatial.distance import cosine
import torch
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import seaborn as sns
from textstat import flesch_reading_ease
import jsonlines
from transformers import DistilBertModel, DistilBertTokenizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
distilbert_model.to(device)


def extract_ideas(text):
    text = text.lower()
    ideas = []

    lines = text.split("\n")
    for line in lines:
        line = line.strip()

        if re.match(r'\d+\.', line):
            idea = re.sub(r'\d+\.', '', line).strip()
            ideas.append(idea)

        elif re.match(r'-', line):
            idea = re.sub(r'-', '', line).strip()
            ideas.append(idea)

        else:
            ideas.append(line)

    ideas = list(set([idea for idea in ideas if idea]))

    return ideas


def load_results(file_path):
    with jsonlines.open(file_path, mode='r') as reader:
        data = [row for row in reader]
    return pd.DataFrame(data)


def bert_embedding(texts, model, tokenizer, batch_size=32):
    try:
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**tokens)

            embeddings = outputs.last_hidden_state
            mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.shape).float()
            masked_embeddings = embeddings * mask
            summed = torch.sum(masked_embeddings, 1)
            lengths = torch.sum(mask, 1)
            average_embeddings = summed / lengths
            embeddings_list.extend(average_embeddings.squeeze().cpu().numpy())
        return embeddings_list
    except Exception as e:
        print(e)
        return np.NaN


# def semantic_embedding(texts, model):
#     embeddings_list = []
#     for text in texts:
#         idea_embeddings = model.encode(text)
#         embeddings_list.append(idea_embeddings)
#     return embeddings_list

def pos_distributions(texts):
    try:
        pos_dists = []
        for text in texts:
            tokens = word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            pos_counts = Counter(tag for word, tag in pos_tags)
            total_tags = sum(pos_counts.values())
            pos_dist = {tag: count / total_tags for tag, count in pos_counts.items()}
            pos_dists.append(pos_dist)
        return pos_dists
    except Exception as e:
        print(e)
        return np.NaN


def average_length(texts):
    try:
        return np.mean([len(text.split()) for text in texts])
    except Exception as e:
        print(e)
        return np.NaN


def average_readability(texts):
    return np.mean([flesch_reading_ease(text) for text in texts])


def cosine_distances(row):
    try:
        distances = []
        outputs = np.mean(row['output_bert_embeddings'], axis=1)
        examples = np.mean(row['example_bert_embeddings'], axis=1)
        distance = cosine(list(outputs), list(examples))
        return distance
    except Exception as e:
        print(e)
        return np.NaN


def pos_distances(row):
    try:
        pd1 = row['output_pos_dists']
        pd2 = row['example_pos_dists']

        # Union of keys in both POS distributions
        all_tags = set(pd1.keys()) | set(pd2.keys())

        # Fill missing tags with zeros
        pd1_vec = [pd1.get(tag, 0) for tag in all_tags]
        pd2_vec = [pd2.get(tag, 0) for tag in all_tags]

        # Calculate cosine distance
        distance = cosine(pd1_vec, pd2_vec)

        return distance
    except Exception as e:
        print(e)
        return np.NaN


def length_distances(row):
    try:
        l1 = np.mean([len(x) for x in row['output']])
        l2 = np.mean([len(x) for x in row['examples']])
        return abs(l2 - l1)
    except Exception as e:
        print(e)
        return np.NaN


def readability_distances(row):
    try:
        output_str = " ".join(row['output'])
        example_str = " ".join(row['examples'])
        return abs(flesch_reading_ease(output_str) - flesch_reading_ease(example_str))
    except Exception as e:
        print(e)
        return np.NaN


def cosineb_distances(row):
    try:
        distances = []
        outputs = row['output_bert_embeddings']
        examples = row['example_bert_embeddings']
        distances = []
        for o_embedding in outputs:
            for e_embedding in examples:
                distance = cosine(list(o_embedding), list(e_embedding))
                distances.append(distance)
        return np.mean(distances)
    except Exception as e:
        print(e)
        return np.NaN


# Function to parallelize
# Function to parallelize
def parallel_processing(df):
    df['output'] = df['output_responses'].apply(lambda x: extract_ideas(x))

    # Semantic embeddings
    # Semantic embeddings
    df['output_bert_embeddings'] = df['output'].apply(
        lambda x: bert_embedding(x, distilbert_model, distilbert_tokenizer))
    df['example_bert_embeddings'] = df['examples'].apply(
        lambda x: bert_embedding(x, distilbert_model, distilbert_tokenizer))

    # Compute POS distributions
    df['output_pos_dists'] = pos_distributions(df['output'].apply(' '.join))
    df['example_pos_dists'] = pos_distributions(df['examples'].apply(' '.join))

    # Compute length and readability
    df['output_length'] = df['output'].apply(average_length)
    df['example_length'] = df['examples'].apply(average_length)
    df['output_readability'] = df['output'].apply(average_readability)
    df['example_readability'] = df['examples'].apply(average_readability)

    # Get distances
    df['cosine_distances'] = df.apply(cosineb_distances, axis=1)
    df['pos_distances'] = df.apply(pos_distances, axis=1)
    df['length_distances'] = df.apply(length_distances, axis=1)
    df['readability_distances'] = df.apply(readability_distances, axis=1)

    return df

def main():

    # Load data
    res = load_results("results_2023-04-02-16-38-06.jsonl").sample(10)

    # Split the DataFrame into equal chunks for parallel processing
    num_chunks = os.cpu_count()
    data_chunks = np.array_split(res, num_chunks)

    # Run parallel processing
    with ProcessPoolExecutor(num_chunks) as executor:
        results = executor.map(parallel_processing, data_chunks)

    # Load data
    res = load_results("results_2023-04-02-16-38-06.jsonl")

    # Split the DataFrame into equal chunks for parallel processing
    num_chunks = os.cpu_count()
    data_chunks = np.array_split(res, num_chunks)

    # Run parallel processing
    with ProcessPoolExecutor() as executor:
        results = executor.map(parallel_processing, data_chunks)

    # Concatenate the results back into a single DataFrame
    res = pd.concat(results)
    res.to_json("final_pull.json", orient='records')
    res.to_csv("final_pull.csv")

if __name__ == "__main__":
    main()
