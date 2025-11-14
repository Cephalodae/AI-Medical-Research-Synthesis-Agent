from flask import Flask, render_template, request, redirect, url_for
import tempfile
#from rag import FUNCTION HERE
import os
import Medical_Agent_AI

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def documents_page():
    return render_template('main.html')

@app.route('/results', methods=['POST'])
def results_page():
    pdfs = request.files.getlist('input_docu')
    prompt = request.form.get('prompt')

    pdf_paths = []
    ai_response = "error"

    with tempfile.TemporaryDirectory() as pdf_loc:
        for pdf in pdfs:
            if pdf and pdf.filename:
                path = os.path.join(pdf_loc, pdf.filename)
                pdf.save(path)
                pdf_paths.append(path)

        print(f"Debug: {os.path.exists(pdf_paths[0])}") # debug statement


        #ai_response = CALL rag.py HERE!
        # calling mine instead lol, Medical Agent AI
        if pdf_paths:
            # first try to create the brain of this entire op
            chain = Medical_Agent_AI.create_rag_chain(pdf_paths)

            # then ask the question
            if chain:
                result = chain.invoke(prompt)
                ai_response = result.content # take the text only
            else:
                ai_response = "Error, could you try again? Couldn't read the PDFs"

        # the document sadly gets deleted at this point :(
        return render_template('results.html', ai_response=ai_response)

# if __name__ == '__main__':
#     app.run()
