from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route("/")
def consent_form():
    return render_template('consent_form.html')

@app.route("/experiment", methods=['GET', 'POST'])
def experiment():
    if request.method == 'POST':
        # Do something with the participant's response
        response = request.form.get('response')
        # Redirect the user to the thank you page
        return redirect(url_for('thank_you'))
    else:
        # Render the experiment page
        return render_template('experiment.html')



@app.route("/thank-you")
def thank_you():
    return render_template('thank_you.html')

if __name__ == '__main__':
    app.run(port=5001, debug=True)
