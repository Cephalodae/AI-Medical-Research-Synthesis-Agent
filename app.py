import fitz
from flask import Flask, render_template, request, redirect, url_for
import time

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def documents_page():
    return render_template('main.html')

@app.route('/results', methods=['POST'])
def results_page():
    pdfs = request.files.getlist('input_docu')

    extracted_text = ""
    for pdf in pdfs:
        pdf_convert = pdf.read()

        fitz_pdf = fitz.open(stream=pdf_convert)
        for page in fitz_pdf:
            extracted_text += page.get_text()
        fitz_pdf.close()

    return render_template('results.html', extracted_text=extracted_text)

if __name__ == '__main__':
    app.run()
