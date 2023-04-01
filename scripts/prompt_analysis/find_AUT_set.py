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


def main(N=4):
    log_file = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode="w", format='%(asctime)s %(message)s')
    logging.info("number of objects: {}".format(N))

    # Read in AUT scores that were used in the paper:
    # Dumas, D., Organisciak, P., & Doherty, M. (2021). Measuring divergent thinking originality with human raters and text-mining models: A psychometric comparison of methods. Psychology of Aesthetics, Creativity, and the Arts, 15(4), 645–663.
    # Note: The link to the data was obtained by Joshua Ashkinaze contacting Peter Organisciak

    # Check if the data file already exists
    dest_path = "../../data/dumas_et_al.csv"
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
                                         position = "h!",
                                         caption='Summary statistics of originality scores (1-5) from \citet{dumas_measuring_2021}',
                                         label='tab:prior_work_prompts'))

    # Get combo with smallest sd
    means = df.groupby(by=['prompt']).mean().reset_index()[['prompt', 'human_vote']]
    all_combos = list(itertools.combinations(means['prompt'].tolist(), N))
    data = []
    for combo in all_combos:
        combo_var = np.std(means[means['prompt'].isin(combo)]['human_vote'].tolist())
        data.append({'combo': str(combo), 'sd': combo_var})
    combo_df = pd.DataFrame(data)
    combo_df = combo_df.sort_values(by=['sd'], ascending=True)
    logging.info("Head of DF")
    logging.info(combo_df.sort_values(by=['sd']).head(5))
    logging.info("Most Sim Subset")
    sim_subset = combo_df.sort_values(by=['sd']).head(1)
    logging.info(sim_subset)

    prompts = sim_subset['combo'].tolist()[0].replace('(', '').replace(')', '').replace("'", '').split(', ')

    # Get a sample of responses from this subset that are mid-originality
    # (i.e: 25% to 75% percentile)
    sample_df = df[df['prompt'].isin(prompts)] \
        .groupby('prompt') \
        .apply(lambda x: x[(x['human_vote'] >= x['human_vote'].quantile(0.25)) &
                           (x['human_vote'] <= x['human_vote'].quantile(0.75))] \
               .drop_duplicates(subset=['response']) \
               .sample(n=10, random_state=56)) \
        .reset_index(drop=True)[['prompt', 'response']]
    sample_df.to_csv(f"../data/{N}_subset_samples.csv", index=False)


if __name__ == "__main__":
    main()
