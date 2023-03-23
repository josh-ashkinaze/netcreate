""""
Date: 03/12/2023

Description: This is a Flask web application that runs a web experiment.

* Participants go through a series of trials, where they are asked to respond to various prompts under different conditions.
* The application collects participants' responses and stores them in a Google BigQuery database.
* The experiment counterbalances across different conditions and stimuli
* The experiment has four routes:  the consent form, the start of the experiment, the trials themselves, and the thank you page.
"""

# TODO add GPT prompts

from flask import Flask, render_template, request, redirect, url_for
from google.cloud import bigquery
from google.oauth2 import service_account
import uuid
import time
from datetime import datetime
import random

# EXPERIMENT PARAMETERS
####################
CONDITIONS = {
    'hh': {'source': 'human', 'label': 'humans'},
    'ha': {'source': 'human', 'label': 'artificial intelligence'},
    'aa': {'source': 'ai', 'label': 'artificial intelligence'},
    'ah': {'source': 'ai', 'label': 'humans'}
}
ITEMS = ['a book', 'a bottle', 'a fork', 'a tire']

N_EXAMPLES = 5
####################


# INIT DICT TO KEEP TRACK OF INFO FOR EACH PARTICIPANT
TEMP = {
    "item_order": None, # Order of items
    "condition_order": None, # Order of experimental conditions
    "participant_id": None, # Unique participant ID
}

app = Flask(__name__)

key_path = "../creds/netcreate-0335ce05e7ff.json"
credentials = service_account.Credentials.from_service_account_file(
    key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

client = bigquery.Client(credentials=credentials, project=credentials.project_id)
dataset = client.dataset("net_expr")
table = dataset.table("trials")


@app.route("/")
def consent_form():
    """Consent form"""
    return render_template('consent_form.html')


@app.route("/start-experiment")
def start_experiment():
    """
    Start the experiment.

    Let's first create a participant ID, counterbalance conditions, and counterbalance items.
    Store these items in the global variable called `TEMP'.

    Then, we start the experiment by calling `render_trial(conditon_no=0)'.
    From there, `render_trial' will recursively call itself to display all trials until the experiment is done.

    """
    TEMP['participant_id'] = str(uuid.uuid4())
    item_order = list(ITEMS)
    random.shuffle(item_order)
    condition_order = list(CONDITIONS.keys())
    random.shuffle(condition_order)
    TEMP['condition_order'] = condition_order
    TEMP['item_order'] = item_order
    return redirect(url_for('render_trial', condition_no=0, method="GET"))


@app.route("/render_trial/<int:condition_no>", methods=['GET', 'POST'])
def render_trial(condition_no):
    """
    Recursively handles the render_trial route for a particular condition_no.

    The idea is that in a temp dictionary, we store the participant ID, the condition order, and the item order.
    Then, we keep calling this function with the next condition_no -- which indexes items and conditions --
    until we've gone through all conditions.

    The logic is as follows:

    IF the current condition_no number is more than the number of items:
        Return the thank_you page since our experiment is done.
    
    ELSE if there are still trials to go:
        1. If the HTTP method is GET (i.e: response not submitted), retrieve the necessary context from the global TEMP
        dict and generate an render_trial instance. Upon submitting a response, this submits a post request.

        2. If the HTTP method is POST (i.e: response was submitted), the function retrieves the participant's response
        text and inserts it into a BigQuery table. Then we make GET request to `render_trial(condition_no+1)' to
        go to the next condition/item.


    Parameters:
    - condition_no (int): the current condition_no

    Returns:
    - Either another instance of render_trial or the thank_you page

    """
    # If the participant has completed all condition_nos, redirect to thank you page
    if condition_no > len(ITEMS)-1:
        return redirect(url_for('thank_you'))
    else:
        pass

    # Retrieve the necessary information from the global TEMP variable
    label = CONDITIONS[TEMP['condition_order'][condition_no]]['label']
    source = CONDITIONS[TEMP['condition_order'][condition_no]]['source']
    participant_id = TEMP['participant_id']
    condition = TEMP['condition_order'][condition_no]
    item = TEMP['item_order'][condition_no]

    # If the HTTP method is GET, render the render_trial template
    if request.method == "GET":
        time.sleep(0.1)
        rows = list(client.query(
            f"SELECT response_text FROM `net_expr.trials` WHERE (item = '{item}' AND condition = '{condition}') ORDER BY response_date DESC LIMIT {N_EXAMPLES}").result())
        responses = [row['response_text'] for row in rows]
        print("responses", responses)
        return render_template('render_trial.html', item=item, label=label, rows=responses, condition_no=condition_no)

    # If the HTTP method is POST, insert the participant's response into the BigQuery table
    # then increment the condition_no and redirect to the next render_trial
    elif request.method == 'POST':

        # Retrieve the participant's response
        response_text = request.form.get('participant_response')

        # Insert the participant's response into the BigQuery table
        row = {
            "item": item,
            "response_id": str(uuid.uuid4()),
            "participant_id": participant_id,
            "condition_order": condition_no,
            "response_text": response_text,
            "response_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "condition": condition,
            "source": source,
            "label": label,
        }
        errors = client.insert_rows_json(table, [row])
        if not errors:
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))

        # Redirect to next condition_no
        return redirect(url_for('render_trial', condition_no=condition_no + 1, method="GET"))
    


@app.route("/thank-you")
def thank_you():
    """Thank you page"""
    return render_template('thank_you.html')


if __name__ == '__main__':
    app.run(port=5019, debug=True)
