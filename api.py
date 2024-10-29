from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re
import threading
from queue import Queue
import logging
import torch
import argparse

# Argument Parsing
parser = argparse.ArgumentParser(description="Run the AI Chatbot API")
parser.add_argument('--model_name', type=str, default="Llama-3.2-3B-Instruct", help="Model name to load")
parser.add_argument('--api_key', type=str, default="https://mackerel-striking-unduly.ngrok-free.app", help="API key routing to localhost")
args = parser.parse_args()


def inject_api_url(api_url, js_file_path='ui/js/chatbot.js'):
    with open(js_file_path, 'r') as file:
        js_code = file.read()
    js_code = re.sub(r'(const\s+apiUrl\s*=\s*)["\'].*["\']', f'\\1"{api_url}"', js_code)
    with open(js_file_path, 'w') as file:
        file.write(js_code)


inject_api_url(args.api_key)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

lock = threading.Lock()


def load_model_and_tokenizer():
    with lock:
        logging.info("Loading tokenizer and model...")
        model_name = "Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True,
                                                     torch_dtype=torch.float16)
        return model, tokenizer


model, tokenizer = load_model_and_tokenizer()

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")

# Create a text-generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.6,
    top_p=0.90,
    top_k=30
)

# Using a pre-trained NER model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", device=device)

chunks = [
    ''' Enter Your Chunks Here''']

# Set up embeddings and vector store
logging.info("Setting up embeddings and vector store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})


def get_relevant_chunks(question: str) -> list:
    return retriever.get_relevant_documents(question)


def process_question(question: str):
    relevant_chunks = get_relevant_chunks(question)
    context = " ".join([chunk.page_content for chunk in relevant_chunks])

    # Generate final answer using all relevant chunks
    answer_prompt = f""" Enter your promt to the model here

    Context: {context}

    Question: {question}

    Answer:"""

    final_answer = llm_pipeline(answer_prompt, max_new_tokens=200, num_return_sequences=1)[0]['generated_text']

    # Extract the actual answer from the generated text
    answer_start = final_answer.find("Answer:")
    if answer_start != -1:
        final_answer = final_answer[answer_start + 7:].strip()

    final_answer = final_answer.split("Question:")[0]
    final_answer = final_answer.split("Note:")[0]
    final_answer = final_answer.split("Signature:")[0]
    return final_answer


def get_answer_fine_tuned(question: str, result_queue: Queue):
    try:
        answer = process_question(question)
        result_queue.put(answer)
    except Exception as e:
        result_queue.put(f"Error: {str(e)}")


@app.route('/', methods=['POST'])
def home():
    try:
        data = request.json
        question = data.get('question', '')

        result_queue = Queue()
        thread = threading.Thread(target=get_answer_fine_tuned, args=(question, result_queue))
        thread.start()
        thread.join()

        answer = result_queue.get()

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
