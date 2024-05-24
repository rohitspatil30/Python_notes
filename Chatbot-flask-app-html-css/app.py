
from flask import Flask, render_template, request, send_file
from flask import Flask, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Load the API key from the environment variables
API_KEY = os.getenv('GOOGLE_API_KEY')

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
        You are customer service chatbot. Your name is "CraftBot", whose job is to reply to question asked by the user. 
     Read the document(s), understand what the document(s) is/are all about, understand what are the headings and subheadings given in the document(s), along with it's definition and text under the headings and subheadings. also note that you have to understand all types of fonts, languages, styles, and sizes. No matter how much data the document contains, you have to learn every detail of the context.

     Now read the question carefully and look for the answer in the context provided, by following the given conditions below.
     
     1. If the question asked is direct and straight forward, whose text/context/topic/paragraph/meaning/summary/heading/subheading/nearby-related topic/little off topic but still relevent/code is present in the context, then answer the question considering the document/context OR as per you knowledge and understanding for the topic, and give a proper answer to the question asked.

     2. If the question is indirect or the question asked has no direct reference in the context, but the meaning of the question has an answer from the document/context, that is question related to the document/context, then too answer the question correctly and giving complete information from your side also using your knowledge about the topic and in a proper structure.

     3. If the question is partially related to the document/context, that is (i.e) some part has an answer in the context and other parts have no relevance with the context then you must answer the questions that are related to context provided which has complete relevance in the text provided, and after that at the end reply that (part with no relevance) does not have any answer in the context/document.

     4. If the complete question is unrelated to the context, then reply that "The context provided does not contain any relevent answer about your question, but I can provide you a brief about the query asked using my knowledge: ". and then you can provide an information from your side by applying your knowledge. But do it if it is a geniune attempt by the user to get the information. else reply "Your question is not very relevent to the information I have", and then describe what all can the user ask from what you have understood from the document/contxt provided to you. 

     5 .NOTE AS CHATBOT:
     Apart from the questions and9 it's types, being a chatbot you also have to reply to the questions which are not a question to be answered from context but the question can be in form of usual conversation for chatbot, like "hi", "hello", "bye" etc. you should reply to these questions regardless the context provided, that is without looking for the answer in the context, because as a chatbot your work is not only to answer questions but also to have a conversation with the users.

     NOTE:
     answer all types of questions related to the context ,Example: difference between/ benefits/ features/ main points/ summary/ what do you understand, from the topic of context or related context. 
     
     Long story short answer the question.\n\n\n\n\n\n\n
    Context/Document(s):\n {context}?\n\n\n\n\n\n\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@app.route('/', methods=['GET', 'POST'])
def home():
    pdf_text = None
    if request.method == 'POST':
        pdf_files = request.files.getlist('file')
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        vector_store = get_vector_store(text_chunks)
        pdf_text = raw_text
    return render_template('index.html', pdf_text=pdf_text)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query')
    context = data.get('context')
    print("Received context:", context)  # Print the received context for debugging
    if not context:
        return "Error: Context is empty"
    if not query:
        return "Error: Query is empty"
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(context)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    print("Response:", response)
    return jsonify({'answer': response["output_text"]})


if __name__ == '__main__':
    app.run(debug=True)
