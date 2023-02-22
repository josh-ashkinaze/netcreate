"""
Description: This script finds the set of 5 items with the lowest variance of originality for the AUT task

Author: Joshua Ashkinaze

Date: 2023-02-21
"""


import pandas as pd
import itertools
import logging
import os
import numpy as np
def main():
    log_file = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')

    # Read in AUT scores from
    # https://github.com/massivetexts/open-scoring/tree/master/data
    #
    # These scores are referenced in the paper:
    # Organisciak, P., Acar, S., Dumas, D., & Berthiaume, K. (2022).
    # Beyond Semantic Distance: Automated Scoring of Divergent Thinking Greatly Improves with Large Language Models.
    # https://doi.org/10.13140/RG.2.2.32393.31840
    df = pd.read_csv("../data/AUT_Coding_ColoradoScores_v2.csv")
    df = df.dropna(subset=['Colorado Similarity Scores'])
    df['sim'] = df['Colorado Similarity Scores']
    df = df.query("sim>=0&sim<=1")

    # Now we want to find the set of 5 items with the lowest variance of creativity
    means = df.groupby(by=['Prompt']).mean().reset_index()[['Prompt', 'sim']]
    all_combos = list(itertools.combinations(means['Prompt'].tolist(), 5))
    data = []
    for combo in all_combos:
        combo_var = np.var(means[means['Prompt'].isin(combo)]['sim'].tolist())
        data.append({'combo': str(combo), 'var': combo_var})
    combo_df = pd.DataFrame(data)
    logging.info("Lowest variance subset of 5")
    logging.info(combo_df.sort_values(by=['var'], ascending=True).head(1))
    logging.info(combo_df['var'].describe())


main()