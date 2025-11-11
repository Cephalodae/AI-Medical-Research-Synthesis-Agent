from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def initial_page():
    return render_template('main.html')

@app.route('/documents')
def documents_page():
    return 'Document view'

if __name__ == '__main__':
    app.run()
