import gradio as gr
from dotenv import load_dotenv
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os
import json

# Load environment variables from .env file
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load and preprocess data
file_path = '/home/vivek/Documents/Rahul_Latotra/dataset.json'

def load_and_preprocess_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    def preprocess_job_entry(entry):
        fields_to_include = [
            'Title', 'Company name', 'Job location', 'Work experience', 'Job description',
            'Skills', 'Education details', 'Benefits', 'About company', 'Portal link',
            'Job listing link', "Company's Rating", 'No. of openings', 'Applicants',
            'Job_posting_date', 'Minimum salary', 'Maximum salary', 'Average salary', 'Others'
        ]
        text_data = []
        for field in fields_to_include:
            if field in entry:
                value = entry[field]
                if isinstance(value, list):
                    value = ", ".join([str(item).strip() for item in value if item])
                elif value is None:
                    value = "N/A"
                else:
                    value = str(value).strip()
                if value:
                    text_data.append(f"{field}: {value}")
        return "\n".join(text_data)

    preprocessed_data = [preprocess_job_entry(entry) for entry in data]
    return preprocessed_data

preprocessed_data = load_and_preprocess_data(file_path)

# Setup models and embeddings
embedding_model_name = 'sentence-transformers/all-MiniLM-L12-v2'
llm_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)

# Prompt setup for LLMChain
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant specialized in providing detailed and contextually relevant answers based on job listings data.
    The context provided is sourced from a vector database (FAISS index). Use this information to generate accurate and contextually relevant responses.
    
    When answering the user's query, follow these guidelines:
    1. Only provide job listings that exactly match the user's query, based on job title, location, or specified criteria.
    2. List as many relevant job listings as possible based on the user's query, not just 2 or 3. Keep the listings concise without detailed descriptions, unless explicitly requested.
    3. If no relevant job listings are found, respond succinctly with: 'Based on the context provided, I found no relevant job listings for [role] in [location].' 
    4. Avoid offering general advice, suggestions, or extra information such as how to search for jobs, unless explicitly requested by the user.
    5. Do not generate responses that include roles, companies, or locations that are outside the scope of the user's query.
    6. Keep responses concise, direct, and strictly relevant to the query.
    """),
    ("user", "Context: {context}\n\nUser Query: {question}\n"),
])

llm_endpoint = HuggingFaceEndpoint(
    repo_id=llm_model_name,
    max_length=2000,
    temperature=0.5,
    token=HUGGINGFACEHUB_API_TOKEN
)

llm_chain = LLMChain(llm=llm_endpoint, prompt=prompt_template)

# Helper functions
def truncate_text(text, max_tokens=512):
    tokens = embedding_tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return embedding_tokenizer.convert_tokens_to_string(tokens)

def split_into_chunks(text, max_tokens=512):
    tokens = embedding_tokenizer.tokenize(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [embedding_tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

def generate_embeddings(texts, tokenizer, model, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings

def setup_faiss_index(embeddings):
    embedding_dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(embedding_dim)
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)
    return index

embeddings = generate_embeddings(preprocessed_data, embedding_tokenizer, embedding_model)
index = setup_faiss_index(embeddings)

def chatbot(query, history=[]):
    query_lower = query.lower()

    # Handle common greetings and support queries
    if query_lower in ["hello", "hi", "hey", "greetings"]:
        response = "Hello! How can I assist you today? Are you looking for a specific job listing?"
    elif query_lower in ["help", "support"]:
        response = "I can assist you with queries related to job listings, interview questions, and company information. How can I help?"
    elif query_lower.startswith("navigate"):
        response = "Sure! Please tell me what specific information or section you'd like to navigate to, such as job descriptions, company details, etc."
    else:
        # Correct query if needed
        corrected_query = auto_correct(query)
        if corrected_query != query:
            response = f"I corrected your query to: {corrected_query}. Here's what I found:"
            query = corrected_query
        else:
            response = ""

        # Embed and search for relevant job listings
        inputs = embedding_tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            query_embedding = embedding_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().astype('float32')

        k = 30  # Search for up to 10 jobs to have more results
        distances, indices = index.search(query_embedding, k)
        retrieved_entries = [preprocessed_data[i] for i in indices[0] if i < len(preprocessed_data)]

        # Check if the user is asking for the total number of jobs
        if "how many" in query_lower or "total" in query_lower:
            total_jobs = len(retrieved_entries)
            response = f"I found {total_jobs} jobs that match your query."

        elif not retrieved_entries or all(entry.strip() == "" for entry in retrieved_entries):
            # If no relevant listings are found, provide a general, LLM-based response
            user_query = truncate_text(query, 512)
            response = llm_chain.run({
                "context": "No job listings were found.",
                "question": user_query
            })
        else:
            # Limit each job listing to a few lines to prevent overly long responses
            retrieved_entries = [entry[:300] for entry in retrieved_entries]  # Limit each entry to 300 characters
            context = "\n\n".join(retrieved_entries)

            # If the context is still too long, chunk it
            context_chunks = split_into_chunks(context, 2000)  # Split context into chunks
            user_query = truncate_text(query, 512)

            # Generate the response for the first chunk
            context_info = context_chunks[0]
            response = llm_chain.run({
                "context": context_info,
                "question": user_query
            })

            # If there are more chunks, handle them in subsequent responses
            if len(context_chunks) > 1:
                response += "\n\nThere are more results. Please refine your query or ask for additional listings."

    history.append((query, response))
    return history, history

def auto_correct(query):
    corrections = {
        "helo": "hello",
        "im": "I'm",
        "recieve": "receive",
        "teh": "the"
    }
    # Correct each word in the query if it exists in the corrections dictionary
    corrected_query = " ".join([corrections.get(word, word) for word in query.split()])
    return corrected_query

# Gradio interface setup
with gr.Blocks() as demo:
    gr.Markdown("## Job Listings Chatbot 🧑‍💼💬")
    gr.Markdown("Ask me anything about job listings!")
    chatbot_output = gr.Chatbot()
    user_input = gr.Textbox(label="Enter your query:", placeholder="Type your job-related question here...")
    state = gr.State([])

    def submit(query, history):
        history, output = chatbot(query, history)
        return output, history

    user_input.submit(submit, [user_input, state], [chatbot_output, state])
    gr.Markdown("**Note:** Ensure the dataset.json file is correctly formatted.")

demo.launch(debug=True, share=True)
