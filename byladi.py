import os
import uuid
import json
import base64
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from byaldi import RAGMultiModalModel
import anthropic

app = Flask(__name__)

# Initialize the RAG model with the correct version
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# File to store document IDs and their corresponding index paths
DOCUMENT_INDEX_FILE = '/runpod-volume/document_indices.json'

UPLOAD_FOLDER = '/runpod-volume/uploads'
INDEX_FOLDER = '/runpod-volume/indices'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(INDEX_FOLDER):
    os.makedirs(INDEX_FOLDER)

# Load existing document indices
def load_document_indices():
    if os.path.exists(DOCUMENT_INDEX_FILE):
        with open(DOCUMENT_INDEX_FILE, 'r') as f:
            return json.load(f)
    return {}

# Save document indices
def save_document_indices(indices):
    with open(DOCUMENT_INDEX_FILE, 'w') as f:
        json.dump(indices, f)

# Initialize document_indices
document_indices = load_document_indices()

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
        index_path = os.path.join(INDEX_FOLDER, f"{index_name}.faiss")
        
        # Index the document
        RAG.index(
            input_path=file_path,
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True
        )
        
        # Debug information
        print(f"Index name: {index_name}")
        print(f"Index path: {index_path}")
        
        # Check if the index was created in the .byaldi directory
        byaldi_index_path = os.path.join('.byaldi', index_name)
        if os.path.exists(byaldi_index_path):
            # Move the index to the desired location
            os.rename(byaldi_index_path, index_path)
            print(f"Moved index from {byaldi_index_path} to {index_path}")
        else:
            print(f"Index not found at {byaldi_index_path}")
            return jsonify({"error": "Failed to create index"}), 500
        
        # Store the document ID and index path
        document_indices[doc_id] = index_path
        save_document_indices(document_indices)
        
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
    
    index_path = document_indices[doc_id]
    
    # Load the specific index for this document from the network volume
    if os.path.exists(index_path):
        RAG_specific = RAGMultiModalModel.from_index(index_path)
    else:
        return jsonify({"error": "Index file not found"}), 500
    
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
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Here are some relevant document excerpts:\n\n"
                }
            ]
        }
    ]

    for idx, result in enumerate(serializable_results, 1):
        messages[0]["content"].extend([
            {"type": "text", "text": f"Excerpt {idx}:\n"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": result['base64']}},
            {"type": "text", "text": f"Metadata: {result['metadata']}\n\n"}
        ])

    messages[0]["content"].append({"type": "text", "text": f"Based on these excerpts, please answer the following question: {query}"})

    claude_response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=4000,
        temperature=0,
        messages=messages
    )
    
    return jsonify({
        "results": serializable_results,
        "answer": claude_response.content[0].text,
        "tokens_consumed": {
            "prompt_tokens": claude_response.usage.input_tokens,
            "completion_tokens": claude_response.usage.output_tokens,
            "total_tokens": claude_response.usage.input_tokens + claude_response.usage.output_tokens
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
        
        # Process the image query with RAG model
        rag_results = RAG.search(query, image_path=image_path)
        
        # Convert Result objects to dictionaries
        serializable_results = [
            {
                "doc_id": result.doc_id,
                "page_num": result.page_num,
                "score": result.score,
                "metadata": result.metadata,
                "base64": result.base64
            } for result in rag_results
        ]
        
        # Encode the query image to base64
        with open(image_path, "rb") as image_file:
            encoded_query_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Clean up the temporary image file
        os.remove(image_path)
        
        # Process results with Claude
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's the query image:"
                    },
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": encoded_query_image}
                    },
                    {
                        "type": "text",
                        "text": "Here are some relevant image results:\n\n"
                    }
                ]
            }
        ]

        for idx, result in enumerate(serializable_results, 1):
            messages[0]["content"].extend([
                {"type": "text", "text": f"Image {idx}:\n"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": result['base64']}},
                {"type": "text", "text": f"Metadata: {result['metadata']}\n\n"}
            ])

        messages[0]["content"].append({"type": "text", "text": f"Based on these images, please answer the following question: {query}"})

        claude_response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0,
            messages=messages
        )
        
        return jsonify({
            "results": serializable_results,
            "answer": claude_response.content[0].text,
            "query_image_base64": encoded_query_image,
            "tokens_consumed": {
                "prompt_tokens": claude_response.usage.input_tokens,
                "completion_tokens": claude_response.usage.output_tokens,
                "total_tokens": claude_response.usage.input_tokens + claude_response.usage.output_tokens
            }
        }), 200
    return jsonify({"error": "Invalid image file"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)