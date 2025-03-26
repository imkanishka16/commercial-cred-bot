from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import chromadb
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel, PrivateAttr
from typing import List
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from functools import wraps

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

class ChromaDBRetriever(BaseRetriever, BaseModel):
    """Enhanced retriever for educational content"""
    _collection: any = PrivateAttr()
    _embedding_function: any = PrivateAttr()
    top_k: int = 15
    min_similarity: float = 0.6

    def __init__(self, **data):
        super().__init__(**data)
        client = chromadb.HttpClient(host='13.232.198.216', port=8000)
        self._collection = client.get_collection(
            name="commercial_credit",
            embedding_function=embedding_function
        )
        self._embedding_function = embedding_function
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        query = self._preprocess_query(query)
        
        results = self._collection.query(
            query_texts=[query],
            n_results=self.top_k,
            include=['documents', 'distances', 'metadatas']
        )
        
        documents = []
        if results['documents'] and results['documents'][0]:
            for doc, distance, metadata in zip(
                results['documents'][0], 
                results['distances'][0], 
                results['metadatas'][0]
            ):
                similarity = 1 / (1 + distance)
                if similarity >= self.min_similarity:
                    documents.append(
                        Document(
                            page_content=doc,
                            metadata={
                                **metadata,
                                "similarity_score": round(similarity, 3)
                            }
                        )
                    )
        
        return sorted(documents, key=lambda x: x.metadata["similarity_score"], reverse=True)
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching"""
        accounting_terms = {
            "define": "definition",
            "what is": "definition",
            "example": "example",
            "calculation": "example",
            "required": "requirement",
            "disclose": "disclosure",
            "collect": "KYC onboarding"
        }
        
        query_lower = query.lower()
        for term, category in accounting_terms.items():
            if term in query_lower:
                return f"{query} {category}"
        return query

def create_rag_chain():
    template = """You are an expert assistant designed to help employees of Commercial Credit and Finance PLC understand and apply the company's policies on Anti-Money Laundering (AML), Countering the Financing of Terrorism (CFT), and Proliferation Financing (PF) as outlined in the provided document. Your responses must be accurate, complete, and based solely on the context from the document.

    Context:
    {context}

    Question: {question}

    Instructions:
    1. Provide answers using only the information explicitly stated in the provided context from the document.
    2. Quote exact wording from the document when providing definitions, objectives, procedures, or requirements (e.g., use quotation marks for direct citations).
    3. If the question relates to a specific section (e.g., KYC, EDD, PEP), identify and prioritize that section’s content, referencing it explicitly (e.g., 'Section 6.2.1') and include all relevant details from it without omission.
    4. If multiple sections are relevant, prioritize the most specific section and supplement with others only if they add necessary detail, ensuring no key points are missed from the primary section.
    5. If a section appears incomplete (e.g., a list ends abruptly), indicate that the context might be truncated and provide the available information, but do not infer missing items.
    6. Do not infer, extrapolate, or add information beyond what is in the context, even if it seems logical.
    7. If the answer is not found in the provided context, state: "The specific information is not found in the provided sections of the document."
    8. Ensure the response directly addresses the employee’s query with precision, avoiding vague or incomplete answers.
    9. For questions requiring a list (e.g., required information, steps), present all items in the order they appear in the document, using numbered or bulleted lists, and do not omit any items unless explicitly irrelevant. If the list appears truncated, note that the context may be incomplete.
    10. Avoid including unrelated details (e.g., transaction processing) unless directly tied to the question.

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0,
        model="gpt-4-0125-preview",
        max_tokens=2000  # Increased to handle longer responses
    )

    retriever = ChromaDBRetriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": True
        }
    )

    return qa_chain

def rag_response(question: str) -> dict:
    """Process a question and return the response with sources"""
    qa_chain = create_rag_chain()

    try:
        result = qa_chain.invoke({"query": question})
        
        answer = result.get('result', '').strip()
        sources = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in result.get('source_documents', [])
        ]

        return {
            "text_answer": answer,
            "sources": sources,
            "success": True
        }
    except Exception as e:
        print(f"Error during chain execution: {str(e)}")
        return {
            "text_answer": str(e),
            "sources": [],
            "success": False,
            "error": str(e)
        }

app = Flask(__name__)
CORS(app)
AUTH_TOKEN = 'Ab@123'

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"error": "Authorization header is missing"}), 401
        if token.startswith('Bearer '):
            token = token[7:]
        if token != AUTH_TOKEN:
            return jsonify({"error": "Invalid authentication token"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/query', methods=['POST'])
@require_auth
def query():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing 'question' in request body",
                "success": False
            }), 400

        question = data['question']
        response = rag_response(question)
        return jsonify({"answer": response["text_answer"]})
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "success": True
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)