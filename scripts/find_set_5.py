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
import subprocess


def main():
    log_file = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')

    # Read in AUT scores that were used in the paper:
    # Dumas, D., Organisciak, P., & Doherty, M. (2021). Measuring divergent thinking originality with human raters and text-mining models: A psychometric comparison of methods. Psychology of Aesthetics, Creativity, and the Arts, 15(4), 645–663.

    # Note: The link to the data was obtained by Joshua Ashkinaze contacting Peter Organisciak

    # Check if the data file already exists
    dest_path = "../data/dumas_et_al.csv"
    if os.path.isfile(dest_path):
        logging.info("Data file already exists. Skipping download step.")
    else:
        # Download the file
        logging.info("Downloading data file...")
        dumas_et_al_link = "https://osf.io/download/u3yv4/"
        subprocess.call(["wget", "-O", dest_path, dumas_et_al_link])
        logging.info("Data file downloaded.")

    # Summary stats
    df = pd.read_csv(dest_path)
    summary = df.groupby(by=['prompt'])['human_vote'].describe()
    summary['count'] = summary['count'].astype(int)
    summary = summary.round(2).reset_index()
    logging.info("\n" + summary.to_latex(index=False,
                           caption='Summary statistics of originality scores (1-5) from \citet{dumas_measuring_2021}',
                           label='tab:prior_work_prompts'))

    # Get combo with smallest variance
    means = df.groupby(by=['prompt']).mean().reset_index()[['prompt', 'human_vote']]
    all_combos = list(itertools.combinations(means['prompt'].tolist(), 5))
    data = []
    for combo in all_combos:
        combo_var = np.var(means[means['prompt'].isin(combo)]['human_vote'].tolist())
        data.append({'combo': str(combo), 'var': combo_var})
    combo_df = pd.DataFrame(data)
    combo_df = combo_df.sort_values(by=['var'], ascending=True)
    logging.info("Head of DF")
    logging.info(combo_df.sort_values(by=['var']).head(5))
    logging.info("Most Sim Subset")
    logging.info(combo_df.sort_values(by=['var']).head(1))


if __name__ == "__main__":
    main()