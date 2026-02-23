from flask import Flask,request,render_template
import os
import PyPDF2 
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy  

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'

@app.route("/")
def matchresume():
    return render_template('index.html')

#helper functions
def extract_text_from_pdf(file_path):
    text = "" 
    with open(file_path,'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
        return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        return file.read()


def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        return ""

    

@app.route("/matcher", methods=['POST'])
def matcher():
    if request.method =='POST':
        job_description = request.form.get("job_description")
        resume_files = request.files.getlist("resumes") ##getting too many resume

        #executing loop as we have too many resumea
        resumes = []
        for resume_file in resume_files:
          filename =   os.path.join(app.config['UPLOAD_FOLDER'],resume_file.filename)
          resume_file.save(filename)
          resumes.append(extract_text(filename))
        
        if not resumes or not job_description:
            return render_template('index.html',message="Please upload resumes and post job..")

        # main part of the projects
        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        print(vectorizer.toarray()) 
        vectors = vectorizer.toarray()
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]
        

        # Get top 3 resumes and their similarity scores
        if len(resumes) == 1:
           score = round(similarities[0] * 100, 2)
           return render_template(
                  "index.html",
                message=f"Match Score: {score}%",
                top_resumes=[],
                similarity_scores=[]
    )

# If multiple resumes uploaded
        else: 
            top_indices = similarities.argsort()[::-1]  # sort all descending
            top_resumes = [resume_files[i].filename for i in top_indices]
            similarity_scores = [round(similarities[i] * 100, 2) for i in top_indices]

            return render_template(
            "index.html",
            message="Top matching resumes:",
            top_resumes=top_resumes,
            similarity_scores=similarity_scores
    )
    return render_template('index.html')

if __name__=="__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run()