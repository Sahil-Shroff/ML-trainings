import pdfplumber
import faiss
import numpy
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from google.colab import drive

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

pdf_path = '../Data/samsung_report.pdf'
text = extract_text_from_pdf(pdf_path)
print("text extraction complete.")

def chunk_text(text, chunk_size=500):
    """Splits text into smaller chunks."""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(text)
print(f"Total chunks created: {len(chunks)}")

def generate_embeddings(chunks):
    """Generates embeddings for each text chunk using a pre-tained sentence transformer"""
    print("Generating embeddings...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    return embeddings, embedding_model

embeddings, embedding_model = generate_embeddings(chunks)
print("Embeddings generation complete.")

def create_faiss_index(embeddings):
    """Creates a FAISS index for fast similarity search."""
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

index = create_faiss_index(embeddings)
chunk_map = {i: chunks[i] for i in range(len(chunks))}
print("FAISS index created.")

def retrive_relevant_chunks(query, index, embedding_model, chunk_map, top_k3=3):
    """Retrieves the top-k relevant chunks for a given query."""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunk_map[i] for i in indices[0] ]
    
def generate_response(query, context):
    """Generate a response using a pre-trained GPT model."""
    model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoModelForCausalLM.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    input_text = f"Context: {context}\n\nQuestion: {query} \n Answer: "
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, Truncation=True)
    outputs = model.generate(**inputs, max_length=700)
    return tokenizer.decode(output[0], skip_special_tokens=True)

while True:
    query = input("\nEnter your query (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        print("Exiting...")
        break

    print("\nRetrieving relevant chunks...")
    relevant_chunks = retrieve_relevant_chunks(query, index, embedding_model, chunk_map)
    context = "\n".join(relevant_chunks)
    print("\nContext:\n", context)

    print("\nGenerating response...")
    response = generate_response(query, context)
    print("\nGenerated Response:\", response)