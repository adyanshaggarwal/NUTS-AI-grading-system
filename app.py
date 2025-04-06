import os
from flask import Flask, request, render_template, send_file, jsonify
import PyPDF2
import pandas as pd
import google.generativeai as genai
from io import StringIO, BytesIO
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True) 

def clean_csv_output(text):
    text = text.replace("```csv", "").replace("```", "").strip()
    return text


def get_file_content(file_path, file_type):
    try:
        if file_type == "pdf":
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return "\n".join(page.extract_text() for page in reader.pages)
        elif file_type == "csv":
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            lines = content.splitlines()
            expected_columns = len(lines[0].split(',')) if lines else 0
            processed_lines = []
            for line in lines:
                parts = line.split(',')
                if len(parts) > expected_columns:
                    parts = parts[:expected_columns-1] + [','.join(parts[expected_columns-1:])]
                processed_lines.append(','.join(parts))
            return '\n'.join(processed_lines)
    except Exception as e:
        raise Exception(f"Error reading {file_type} file: {str(e)}")
    

def grade_answers(api_key, files):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
  
    contents = {
        "course": get_file_content(files['content_pdf'], "pdf"),
        "questions": get_file_content(files['questions_pdf'], "pdf"),
        "answers": get_file_content(files['answerkey_pdf'], "pdf"),
        "replies": get_file_content(files['replies_csv'], "csv")
    }
    
    prompt = f"""
    GRADE STUDENT ANSWERS BASED ON:
    1. COURSE CONTENT: {contents['course']}
    2. QUESTIONS: {contents['questions']}
    3. ANSWER KEY: {contents['answers']}
    4. STUDENT ANSWERS: {contents['replies']}
    
    
    INSTRUCTIONS:
    1. Compare answers against answer key and teacher's feedback in the csv.
    NOTE: Prioritise teacher feedback over answer key and defer to it incase of conflict ! 
    2. Provide feedback (20-60 words) per question
    3. Assign scores (0-5) with only the following columns: feedback_q1,score_q1,feedback_q2,score_q2,...total_score
    4. PRESERVE EXISTING TEACHER FEEDBACK if present
    5. Only adjust scores if they contradict answer key
    6. Output clean CSV without markdown formatting, **Do not put any commas in the feedbacks**. 
    7. Ensure all rows have the same number of columns
    8. Don't include any additional commentary or notes
    """

    
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192
        }
    )
    return clean_csv_output(response.text)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/grade', methods=['POST'])
def grade():
    try:
        # save uploaded files
        files = {
            'content_pdf': save_uploaded_file(request.files['content_pdf'], 'content.pdf'),
            'questions_pdf': save_uploaded_file(request.files['questions_pdf'], 'questions.pdf'),
            'answerkey_pdf': save_uploaded_file(request.files['answerkey_pdf'], 'answers.pdf'),
            'replies_csv': save_uploaded_file(request.files['replies_csv'], 'replies.csv')
        }
        
        # grade answers
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({'success': False, 'error': "API key not configured"}), 500
            
        ai_grading = grade_answers(api_key, files)
        new_grades = pd.read_csv(StringIO(ai_grading))
        
        # remove duplicate column
        new_grades = new_grades.dropna(axis=1, how='all')
        new_grades = new_grades.loc[:,~new_grades.columns.duplicated()]
        
        concatenate = request.form.get('concatenate', 'false').lower() == 'true'
        
        if concatenate:
           
            original = pd.read_csv(files['replies_csv'])
            
            
            existing_columns = set(original.columns)
            columns_to_add = [col for col in new_grades.columns if col not in existing_columns]
            
            final_grades = original.copy()
            for col in columns_to_add:
                final_grades[col] = new_grades[col]
        else:
            final_grades = new_grades

   
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'graded_results.csv')
        final_grades.to_csv(output_path, index=False, encoding='utf-8')

        
        for file_path in files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return jsonify({
            'success': True,
            'download_link': '/download',
            'preview': final_grades.to_html(
                classes='table table-striped',
                index=False,
                border=0,
                justify='left'
            )
        })

    
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


    
@app.route('/download')
def download():
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], 'graded_results.csv'),
        as_attachment=True,
        download_name='graded_results.csv'
    )

def save_uploaded_file(file, filename):
    if not file:
        raise ValueError("No file uploaded")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return file_path

if __name__ == '__main__':
    app.run(debug=True)


