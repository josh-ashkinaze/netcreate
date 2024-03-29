# RAL

import openai
import pandas as pd
import json
import itertools
import os
import sys
import logging
import concurrent.futures
from datetime import datetime
from tenacity import RetryCallState
import jsonlines

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log
)  # for exponential backoff

#####################################
## GLOBAL CONSTANTS ##
with open('../../creds/openai_creds.json', 'r') as f:
    data = json.load(f)
RANDOM_SEED = 416
API_KEY = data['api_key']
example_df = pd.read_csv("../../data/gt_main2.csv")
AUT_ITEMS = ["brick"]
PROMPTS = {
    "zero_shot": "What are some creative uses for [OBJECT_NAME]? The goal is to come up with a creative idea, which is an idea that strikes people as clever, unusual, interesting, uncommon, humorous, innovative, or different. List [N] creative uses for [OBJECT_NAME].",

    "implicit": "What are some creative uses for [OBJECT_NAME]? The goal is to come up with a creative idea, which is an idea that strikes people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Here are example creative uses: [EXAMPLES] Based on the examples, list [N] creative uses for [OBJECT_NAME] that sounds like the examples.",

    "explicit": "What are some creative uses for [OBJECT_NAME]? The goal is to come up with a creative idea, which is an idea that strikes people as clever, unusual, interesting, uncommon, humorous, innovative, or different. Here are example creative uses: [EXAMPLES] Carefully study the examples and their style, then list [N] creative uses for [OBJECT_NAME] that resemble the given examples. Match the style, length, and complexity of the creative ideas in the examples.",
}
#####################################


def handle_prompt(args):
    prompt_base, object_name, examples, n_examples, temperature, frequency_penalty, presence_penalty = args
    prompt = make_prompt(prompt_base, object_name, examples, n_examples)
    response = generate_responses(prompt, temperature, frequency_penalty, presence_penalty)
    return response


def make_prompt(prompt_base, object_name, examples, n_examples):
    prompt = prompt_base.replace("[OBJECT_NAME]", object_name)
    prompt = prompt.replace("[N]", str(n_examples))
    examples = " ".join(['\n- ' + item for item in examples]) + "\n"
    prompt = prompt.replace("[EXAMPLES]", examples)
    return prompt


@retry(wait=wait_random_exponential(multiplier=30, min=1, max=60), stop=stop_after_attempt(30),before_sleep=before_sleep_log(logging, logging.INFO))
def generate_responses(prompt, temperature, frequency_penalty, presence_penalty):
    openai.api_key = API_KEY
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2000,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        temperature=temperature,
    )
    message = response["choices"][0]["text"]
    print(message)
    return message


def get_examples(df, prompt, n_examples, seed=416):
    return df[df['prompt'] == prompt].sample(n_examples, random_state=seed)['response'].tolist()


def split_ideas(x):
    x = x.split("\n")
    x = [(l.replace("- ", "").strip()).lower() for l in x]
    return x


def log_before_sleep(retry_state: RetryCallState):
    logging.info(f"Waiting {retry_state.next_action.sleep} seconds before retrying...")

def main(N_TRIALS_PER_COMBO=1):
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_file = f"{os.path.splitext(os.path.basename(__file__))[0]}_{date_string}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w', format='%(asctime)s %(message)s')
    results_file = f"results_{date_string}.jsonl"

    grid_search = {
        'n_examples': [4],
        'temperature': [0.6, 0.7, 0.8],
        'frequency_penalty': [1, 1.5],
        'presence_penalty': [1, 1.5]
    }


    logging.info(f"n_trials_combo: {N_TRIALS_PER_COMBO}")

    all_combinations = list(itertools.product(
        grid_search['n_examples'],
        grid_search['temperature'],
        grid_search['frequency_penalty'],
        grid_search['presence_penalty'],
    ))
    logging.info(f"grid_search parameters: {grid_search}, len {len(all_combinations)}")
    logging.info(f"prompts: {PROMPTS}")
    logging.info(f"AUT ITEMS: {AUT_ITEMS}")
    total_requests = len(PROMPTS) * len(AUT_ITEMS) * len(all_combinations) * N_TRIALS_PER_COMBO
    logging.info(f"TOTAL REQUESTS: {total_requests}")

    results = []
    condition_counter = 0
    total_counter = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor, jsonlines.open(results_file,
                                                                                          mode='w') as outfile:
        for aut_item in AUT_ITEMS:
            for combination in all_combinations:
                for trial in range(N_TRIALS_PER_COMBO):
                    n_examples, temperature, frequency_penalty, presence_penalty = combination
                    examples = get_examples(example_df, aut_item, n_examples, seed=condition_counter)

                    for prompt_name, prompt_base in PROMPTS.items():
                        args = (
                            prompt_base,
                            "a " + aut_item,
                            examples,
                            n_examples,
                            temperature,
                            frequency_penalty,
                            presence_penalty
                        )
                        future = executor.submit(handle_prompt, args)
                        generated_response = future.result()

                        row = {
                            'prompt_condition': prompt_name,
                            'trial_no': trial,
                            'idx': condition_counter,
                            'examples': examples,
                            'output_responses': generated_response,
                            'n_examples': n_examples,
                            'temperature': temperature,
                            'frequency_penalty': frequency_penalty,
                            'presence_penalty': presence_penalty
                        }
                        results.append(row)
                        total_counter += 1
                        if total_counter % 100 == 0:
                            logging.info(f"{total_counter} of {total_requests}")
                condition_counter += 1

    with jsonlines.open(results_file, mode='w') as writer:
        for row in results:
            writer.write(row)
main()


if __name__ == "__main__":
    if sys.argv[1]:
        main(sys.argv[1])
    else:
        main()