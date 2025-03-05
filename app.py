from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'xlsx', 'csv'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize LLM and embeddings
llm_options = {
    "LLaMA": Ollama(model="llama2"),
    "Gemma": Ollama(model="gemma")
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_document(filepath):
    try:
        if filepath.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
        elif filepath.endswith('.docx'):
            loader = Docx2txtLoader(filepath)
        elif filepath.endswith('.csv') or filepath.endswith('.xlsx'):
            loader = CSVLoader(filepath)
        else:
            raise ValueError("Unsupported file format")
        return loader.load()
    except ImportError as e:
        raise ImportError(f"Required library not installed: {e}")

def process_document(filepath):
    # Load document
    documents = load_document(filepath)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OllamaEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Save vector store for later use
    vectorstore.save_local("vectorstore")
    return vectorstore

@app.route('/')
def index():
    return render_template('index.html', llm_options=list(llm_options.keys()))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the document
        try:
            vectorstore = process_document(filepath)
            return jsonify({"message": "File uploaded and processed successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query')
    selected_llm = data.get('llm')
    
    if not query_text or not selected_llm:
        return jsonify({"error": "Missing query or LLM selection"}), 400
    
    if selected_llm not in llm_options:
        return jsonify({"error": "Invalid LLM selection"}), 400
    
    # Load vector store (assuming it's already processed)
    try:
        embeddings = OllamaEmbeddings()
        vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        return jsonify({"error": f"Failed to load vector store: {str(e)}"}), 500
    
    # Initialize QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_options[selected_llm],
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # Get response
    try:
        response = qa_chain.run(query_text)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Failed to generate response: {str(e)}"}), 500
if __name__ == '__main__':
    app.run(debug=True)