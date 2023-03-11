
# TODO clean up the database. As of now 'trial' is a dummy value. Don't need it  because it's just row anyway
# TODO fix the specific items

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
    'ha': {'source': 'human', 'label': 'artifical intelligence'},
    'aa': {'source': 'ai', 'label': 'artifical intelligence'},
    'ah': {'source': 'ai', 'label': 'humans'}
}
ITEMS = ['a rope', 'a bat', 'a ball', 'a coffee cup']

N_EXAMPLES = 5
####################


# GLOBAL VARIABLES FOR PARTICIPANT AND condition_no
temp = {
    "item_order": None,
    "condition_order": None,
    "participant_id": None,
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
    return render_template('consent_form.html')


@app.route("/start-experiment")
def start_experiment():
    temp['participant_id'] = str(uuid.uuid4())
    item_order = list(ITEMS)
    random.shuffle(item_order)
    condition_order = list(CONDITIONS.keys())
    random.shuffle(condition_order)
    temp['condition_order'] = condition_order
    temp['item_order'] = item_order
    time.sleep(0.05)
    return redirect(url_for('experiment', condition_no=0))


@app.route("/experiment/<int:condition_no>", methods=['GET', 'POST'])
def experiment(condition_no):
    """
    Recursively handles the experiment route for a particular condition_no. 
    
    If the current condition_no number is more than the number of items, returns the thank you page.
    
    Otherwise, this function will:
    
        1. If the HTTP method is GET, the function retrieves the necessary context from the global temp dict and generates an experiment instance. 

        2. If the HTTP method is POST, the function retrieves the participant's response text and inserts it into a BigQuery table. Then we call `experiment(condition_no+1)'


    Parameters:
    - condition_no (int): the current condition_no

    Returns:
    - A Flask response object containing the rendered experiment template with the item, label, previous responses, and condition_no number as context variables.

    """
    # If the participant has completed all condition_nos, redirect to thank you page
    time.sleep(0.1)
    if condition_no > len(ITEMS)-1:
        return redirect(url_for('thank_you'))

    # Retrieve the necessary information from the global temp variable
    label = CONDITIONS[temp['condition_order'][condition_no]]['label']
    source = CONDITIONS[temp['condition_order'][condition_no]]['source']
    participant_id = temp['participant_id']
    condition = temp['condition_order'][condition_no]
    condition_order = condition_no
    item = temp['item_order'][condition_no]
    id = str(uuid.uuid4())

    # If the HTTP method is POST, insert the participant's response into the BigQuery table
    if request.method == 'POST':
        response_text = request.form.get('participant_response')
        row = {
            "item": item,
            "id": id,
            "participant_id": participant_id,
            "condition_no": condition_no,
            "response_text": response_text,
            "response_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "condition": condition,
            "trial": 1,
            "source": source,
            "label": label,
        }
        errors = client.insert_rows_json(table, [row])
        if not errors:
            print(row)
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))

        # Redirect to next condition_no
        return redirect(url_for('experiment', condition_no=condition_no + 1))
    
    # If the HTTP method is GET, render the experiment template
    else:
        rows = list(client.query(f"SELECT response_text FROM `net_expr.trials` WHERE item= '{item}' ORDER BY response_date DESC LIMIT {N_EXAMPLES} ").result())
        responses = [row['response_text'] for row in rows]
        print("responses", responses)
        return render_template('experiment.html', item=item, label=label, rows=responses, condition_no=condition_no)


@app.route("/thank-you")
def thank_you():
    return render_template('thank_you.html')


if __name__ == '__main__':
    app.run(port=5008, debug=True)
