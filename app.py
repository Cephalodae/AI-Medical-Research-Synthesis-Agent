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
    prompt = request.form.get('prompt')

    extracted_text = ""
    for pdf in pdfs:
        pdf_convert = pdf.read()

        fitz_pdf = fitz.open(stream=pdf_convert)
        for page in fitz_pdf:
            extracted_text += page.get_text()
        fitz_pdf.close()

    # GRANITE (temp code below)
    ai_response = "Prompt: \n" + prompt + "\nExtracted text: " + extracted_text

    return render_template('results.html', ai_response=ai_response)

if __name__ == '__main__':
    app.run()
