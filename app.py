from flask import Flask, render_template, request, redirect, url_for
import tempfile
import rag
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def documents_page():
    return render_template('main.html')

@app.route('/results', methods=['POST'])
def results_page():
    pdfs = request.files.getlist('input_docu')
    prompt = request.form.get('prompt')

    pdf_paths = []
    with tempfile.TemporaryDirectory() as pdf_loc:
        for pdf in pdfs:
            if pdf and pdf.filename:
                path = os.path.join(pdf_loc, pdf.filename)
                pdf.save(path)
                pdf_paths.append(path)

    # CALL rag.py HERE!
    ai_response = "Path: " + pdf_paths[0] + "Prompt: \n" + prompt

    return render_template('results.html', ai_response=ai_response)

if __name__ == '__main__':
    app.run()
