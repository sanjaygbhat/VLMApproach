import os
import uuid
import base64
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from byaldi import RAGMultiModalModel
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

app = Flask(__name__)

# Initialize the RAG model with the correct version
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Store document identifiers and their corresponding indices
document_indices = {}

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        doc_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{doc_id}_{filename}")
        file.save(file_path)
        
        # Create a new index for this document
        index_name = f"index_{doc_id}"
        RAG.index(
            input_path=file_path,
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True
        )
        document_indices[doc_id] = index_name
        
        return jsonify({"document_id": doc_id}), 200
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/query', methods=['POST'])
def query_document():
    data = request.json
    doc_id = data.get('document_id')
    query = data.get('query')
    k = data.get('k', 3)
    
    if not doc_id or not query:
        return jsonify({"error": "Missing document_id or query"}), 400
    
    if doc_id not in document_indices:
        return jsonify({"error": "Invalid document_id"}), 400
    
    index_name = document_indices[doc_id]
    
    # Load the specific index for this document
    RAG_specific = RAGMultiModalModel.from_index(index_name)
    
    # Perform the search on the specific index
    results = RAG_specific.search(query, k=k)
    
    # Convert Result objects to dictionaries
    serializable_results = [
        {
            "doc_id": result.doc_id,
            "page_num": result.page_num,
            "score": result.score,
            "metadata": result.metadata,
            "base64": result.base64
        } for result in results
    ]
    
    # Process results with Claude
    claude_prompt = f"{HUMAN_PROMPT} Here are some relevant document excerpts:\n\n"
    for idx, result in enumerate(serializable_results, 1):
        claude_prompt += f"Excerpt {idx}:\n"
        claude_prompt += f"Content: {result['base64']}\n"
        claude_prompt += f"Metadata: {result['metadata']}\n\n"
    
    claude_prompt += f"Based on these excerpts, please answer the following question: {query}{AI_PROMPT}"
    
    claude_response = anthropic.completions.create(
        model="claude-3-sonnet-20240229",
        max_tokens_to_sample=1000,
        prompt=claude_prompt
    )
    
    return jsonify({
        "byaldi_results": serializable_results,
        "claude_answer": claude_response.completion,
        "tokens_consumed": {
            "prompt_tokens": claude_response.usage.prompt_tokens,
            "completion_tokens": claude_response.usage.completion_tokens,
            "total_tokens": claude_response.usage.total_tokens
        }
    }), 200

@app.route('/query_image', methods=['POST'])
def query_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400
    image = request.files['image']
    query = request.form.get('query')
    
    if not query:
        return jsonify({"error": "Missing query"}), 400
    
    if image and image.filename != '':
        filename = secure_filename(image.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(image_path)
        
        # Process the image query with Byaldi
        byaldi_results = RAG.search(query, image_path=image_path)
        
        # Convert Result objects to dictionaries
        serializable_results = [
            {
                "doc_id": result.doc_id,
                "page_num": result.page_num,
                "score": result.score,
                "metadata": result.metadata,
                "base64": result.base64
            } for result in byaldi_results
        ]
        
        # Encode the query image to base64
        with open(image_path, "rb") as image_file:
            encoded_query_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Clean up the temporary image file
        os.remove(image_path)
        
        # Process results with Claude
        claude_prompt = f"{HUMAN_PROMPT} Here's the query image encoded in base64: {encoded_query_image}\n\n"
        claude_prompt += "Here are some relevant image results:\n\n"
        for idx, result in enumerate(serializable_results, 1):
            claude_prompt += f"Image {idx}:\n"
            claude_prompt += f"Content: {result['base64']}\n"
            claude_prompt += f"Metadata: {result['metadata']}\n\n"
        
        claude_prompt += f"Based on these images, please answer the following question: {query}{AI_PROMPT}"
        
        claude_response = anthropic.completions.create(
            model="claude-3-sonnet-20240229",
            max_tokens_to_sample=1000,
            prompt=claude_prompt
        )
        
        return jsonify({
            "byaldi_results": serializable_results,
            "claude_answer": claude_response.completion,
            "query_image_base64": encoded_query_image,
            "tokens_consumed": {
                "prompt_tokens": claude_response.usage.prompt_tokens,
                "completion_tokens": claude_response.usage.completion_tokens,
                "total_tokens": claude_response.usage.total_tokens
            }
        }), 200
    return jsonify({"error": "Invalid image file"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)