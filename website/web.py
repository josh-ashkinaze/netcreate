from flask import Flask, render_template, request, redirect, url_for
from google.cloud import bigquery
from google.oauth2 import service_account
import uuid
from datetime import datetime
import random

# EXPERIMENT CONDITIONS
####################
CONDITIONS = {
    'hh':{'source':'human', 'label':'humans'},
    'ha':{'source':'human', 'label':'artifical intelligence'},
    'aa':{'source':'ai', 'label':'artifical intelligence'},
    'ah':{'source':'ai', 'label':'humans'}
}
ITEMS = ['a rope', 'a bat', 'a ball', 'a coffee cup']
####################


# GLOBAL VARIABLES FOR PARTICIPANT AND TRIAL
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
    for i in range(len(ITEMS)):
        temp['condition_order'] = condition_order
        temp['item_order'] = item_order
        return redirect(url_for('experiment', trial=0))
    return redirect(url_for('thank_you'))


@app.route("/experiment/<int:trial>", methods=['GET', 'POST'])
def experiment(trial):
    print("TEMP")
    print(temp)
    # Retrieve the necessary information from the global temp variable
    if trial <= 3:
        label = CONDITIONS[temp['condition_order'][trial]]['label']
        source = CONDITIONS[temp['condition_order'][trial]]['source']
        participant_id = temp['participant_id']
        condition = temp['condition_order'][trial]
        condition_order = trial
        item = temp['item_order'][trial]
        id = str(uuid.uuid4())
    else:
        return render_template("thank_you.html")

    if request.method == 'POST':
        # Get the response text and the datetime of the response
        response_text = request.form.get('participant_response')

        # Insert the response into the trials table
        row = {
            "item": item,
            "id": id,
            "participant_id": participant_id,
            "trial": trial,
            "response_text": response_text,
            "response_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "condition": condition,
            "condition_order": condition_order,
            "source": source,
            "label": label,
        }
        errors = client.insert_rows_json(table, [row])
        if errors == []:
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))
        if trial > 3:
            return redirect(url_for('thank_you'))
        else:
            temp['trial'] = trial
            temp['item'] = random.choice(ITEMS)
            temp['condition_info'] = random.choice(list(CONDITIONS.keys()))
            return redirect(url_for('experiment', trial=trial))

    # Generate a prompt for the participant to respond to
    prompt = f"Please describe a novel use for {item}."
    rows = client.query(f"SELECT response_text FROM `net_expr.trials` WHERE participant_id='{participant_id}' and item='{item}'").result()
    responses = [row['response_text'] for row in rows]
    print("THIS PART")
    return render_template('experiment.html', item=item, label=label, rows=responses, trial=trial)

@app.route("/thank-you")
def thank_you():
    return render_template('thank_you.html')

if __name__ == '__main__':
    app.run(port=5008, debug=True)



