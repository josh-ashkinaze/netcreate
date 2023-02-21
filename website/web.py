from flask import Flask, render_template, request, redirect, url_for
from google.cloud import bigquery
from google.oauth2 import service_account
import uuid
from datetime import datetime

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

@app.route("/experiment", methods=['GET', 'POST'])
def experiment():
    if request.method == 'POST':
        # Generate a unique id for the response
        response_id = str(uuid.uuid4())
        # Generate a unique participant id
        participant_id = str(uuid.uuid4())
        # Get the response text and the datetime of the response
        response_text = request.form.get('response')
        # Insert the response into the trials table
        row = {
            "id": response_id,
            "participant_id": participant_id,
            "trial": 1,
            "response_text": response_text,
            "response_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "condition": "1",
            "condition_order": 1,
            "source": "human"
        }
        errors = client.insert_rows_json(table, [row])
        if errors == []:
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))
        # Redirect the user to the thank you page
        return redirect(url_for('thank_you'))
    else:
        # Retrieve all rows from the table and pass them to the template
        query = "SELECT * FROM net_expr.trials"
        rows = list(client.query(query).result())
        # Limit the rows to 8 if there are more than 8, else keep all rows
        num_rows = min(len(rows), 8)
        rows = rows[:num_rows]
        return render_template('experiment.html', rows=rows)

@app.route("/thank-you")
def thank_you():
    return render_template('thank_you.html')

if __name__ == '__main__':
    app.run(port=5001, debug=True)
