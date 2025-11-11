from flask import Flask, render_template
import time

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def documents_page():
    return render_template('main.html')

@app.route('/results', methods=['GET', 'POST'])
def results_page():
    return render_template('results.html')

if __name__ == '__main__':
    app.run()
