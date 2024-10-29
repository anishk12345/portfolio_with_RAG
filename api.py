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
    # Work Experience Summary
    "Work Experience Summary : Before beginning my postgraduate studies at Conestoga College, I accumulated 3 years and 11 months of professional experience across three organizations: \n"
    "1) Intellect Design Arena Ltd. (Chennai, India): Consultant - Data Scientist (June 2021 - December 2022), Third organization i worked for. where I developed Table Structure Recognition systems using U-Net and Faster R-CNN. \n"
    "2) TVS NEXT (Chennai, India): Second organization i worked for. Machine Learning Engineer (April 2020 - May 2021), leading projects such as the Applicant Tracking System, Acronym Classifier, and iOS object detection. \n"
    "3) Rakuten (Bangalore, India): Data Engineer - Support (April 2019 - April 2020). First organization i worked for. I was the youngest on my team, working closely with experienced engineers on critical projects to manage Rakuten's data platform using Hadoop, Spark, and AWS. \n"
    "In total, I have nearly four years of hands-on experience in data science, machine learning, and data engineering roles.",

    # Current Academic Status
    "Education : I completed a postgraduate program in Applied AI and Machine Learning at Conestoga College, graduating in April 2024. Prior to this, I also completed a Predictive Analytics program at Conestoga, which provided a strong foundation in data modeling, statistical analysis, and visualization. Additionally, I hold a Post Graduate Program in Data Science and Engineering from Great Lakes Institute of Management (July 2018 - March 2019), where I specialized in machine learning, data visualization, and advanced analytics. My undergraduate studies were completed at Sathyabama Institute of Science and Technology, where I earned a degree in Computer Science and Engineering (June 2014 - June 2018). These academic experiences have equipped me with a comprehensive skill set in AI, data science, and engineering, which I continue to refine through personal projects."

    # Skills
    "professional Skills Overview: Anish has expertise in data science and machine learning, with hands-on experience in deep learning, NLP, computer vision and LLM's. His technical skills include Python, SQL, TensorFlow, PyTorch, LangChain, MLFlow, Docker, and AWS, as well as Transformers and knowledge of both SQL and NoSQL databases. Anish has also developed systems for large-scale data management and is continually expanding his expertise in emerging AI and ML technologies."

    # Awards
    "Awards : Recognized with multiple 'Best Performer' awards during my time at TVS NEXT and Intellect Design Arena Ltd., awarded for exceptional contributions to project outcomes and successful implementations of complex data science solutions.",

    # Certifications
    "Certifications : Certified in Big Data by FITA Academy, and in Cloud Computing from Sathyabama Institute of Science and Technology, enhancing my expertise in managing large-scale data systems and cloud-based platforms.",

    # Contact Information
    "Contact Information : Email: krishnananish24@gmail.com, Phone: +1 (647)-939-9682. You can also reach out through the Contact Page on my website.",

    # Key Past Projects
    "Key Past Projects : In my previous roles, I contributed to the following projects that involved real-world applications of data engineering and machine learning:\n"
    "1) **Rakuten Migration Project**: In this project, I worked on performance optimization, incident resolution, and ensuring smooth data platform migration. My responsibilities included managing job failures, coordinating resolutions, and using monitoring tools like Nagios to maintain system stability during high-traffic events. I optimized Hadoop and Spark data pipelines and ensured data processing continuity across Rakuten’s AWS infrastructure.\n\n"
    "2) **Acronym Classifier**: Developed to extract and expand abbreviations in documents, particularly in legal contexts, this project combined a custom extraction method with the Blackstone NLP model to achieve high accuracy. The system leverages regex heuristics and NLP techniques for expansion prediction and is deployed in Docker to support real-time document processing.\n\n"
    "3) **Applicant Tracking System (ATS)**: This ATS automates resume parsing and candidate ranking with NLP and Named Entity Recognition (NER) modules. It identifies and extracts skills, roles, and qualifications from resumes, using similarity scoring to match candidates to job descriptions. The system assists recruiters by providing match scores that streamline candidate selection.\n\n"
    "4) **Table Structure Recognition**: This system extracts structured data from both bordered and borderless tables within images. For bordered tables, I implemented a U-Net model to recognize cell boundaries. For borderless tables, a Faster R-CNN with ResNet-101 was used to identify text and spacing patterns. Post-processing techniques, including bounding box merging, handle complex structures such as rowspan and colspan, providing structured datasets for easy analysis.",

    # Personal Projects
    "Personal and Current Projects: Alongside my studies, I am working on personal projects to expand my technical expertise. Some of these include:\n"
    "- **AI-Powered Chatbot**: I developed a chatbot for my portfolio website using webflow, designed to answer questions about my professional background and skill set. This chatbot combines LangChain with a Retrieval-Augmented Generation (RAG) system, leveraging natural language understanding and llama3.2 models to provide accurate and context-aware responses.\n\n"
    "- **SQL Automation Project**: This project generates SQL queries using an llm to automate the resolution of common query failures. It leverages knowledge graphs to gather context about the database and related queries, enabling smarter and more accurate query generation. Additionally, it automates ticket creation in Jira for unresolved issues, streamlining the process of handling simpler failures and reducing the need for manual intervention in incident resolution. This project is being work on currently"
]

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
    answer_prompt = f"""Answer the following question based strictly on the provided context. Respond concisely, 
    and only include the requested information without adding unrelated details, signatures, or formal notes. Only 
    talk about the technical details when asked about the projects. Assume Anish is answering the questions himself, 
    using first-person language. If asked about technical skills not mentioned in the skill list, state that Anish is 
    curious and continuously learning new technologies. For off-topic questions not related to Anish’s professional 
    work, give a brief professional reason and decline to answer. Only answer with facts from the provided context. 
    Do not add extra context, personal notes, or assumptions. Use professional structured answers with proper 
    punctuation. When asked about a project give brief explanation about the project. If talking about organizations 
    always mention the name of the organization.

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
