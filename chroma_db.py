# # chroma_db.py
# import chromadb
# from pypdf import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# import os
# from dotenv import load_dotenv
# from typing import List, Dict
# import re
# import logging

# # Initialize logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# load_dotenv()

# EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Using consistent model with 384 dimensions
# embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

# def clean_text(text: str) -> str:
#     """Clean and normalize text while preserving educational content structure"""
#     text = re.sub(r'(?<=\n)\s*(\d+\.|\•|\-)\s*', r'\n\1 ', text)  # Preserve numbered lists and bullets
#     text = re.sub(r'\n([A-Z][A-Za-z\s]+)(?=\n)', r'\n### \1', text)  # Preserve section headers
#     text = re.sub(r'[^\w\s.,;:?!()\[\]{}\-–—•\n\*]', '', text)  # Clean unwanted characters
#     return text.strip()

# def detect_content_type(text: str) -> str:
#     """Detect type of educational content"""
#     text_lower = text.lower()
#     if "definition" in text_lower or "means" in text_lower:
#         return "definition"
#     elif "example" in text_lower:
#         return "example"
#     elif any(word in text_lower for word in ["shall", "must", "required"]):
#         return "requirement"
#     elif "disclosure" in text_lower:
#         return "disclosure"
#     return "explanation"

# def get_pdf_text(pdf_dir: str) -> List[Dict[str, str]]:
#     """Extract text with enhanced educational content structure, skipping image-only pages"""
#     documents = []
    
#     for file_name in os.listdir(pdf_dir):
#         if not file_name.endswith(".pdf"):
#             continue
            
#         file_path = os.path.join(pdf_dir, file_name)
#         try:
#             pdf_reader = PdfReader(file_path)
#         except Exception as e:
#             logger.error(f"Failed to read {file_name}: {e}")
#             continue
        
#         standard_match = re.search(r'([A-Z]+)\s*(\d+)', file_name)
#         standard_info = {
#             "standard": standard_match.group(1) if standard_match else "",
#             "number": standard_match.group(2) if standard_match else ""
#         }
        
#         current_section = ""
#         for page_num, page in enumerate(pdf_reader.pages):
#             text = page.extract_text() or ""  # Default to empty string if None
#             if text.strip():  # Only process pages with extractable text
#                 section_match = re.search(r'\n([A-Z][A-Za-z\s]+)(?=\n)', text)
#                 if section_match:
#                     current_section = section_match.group(1)
                
#                 cleaned_text = clean_text(text)
#                 logger.debug(f"Page {page_num + 1} Cleaned Text: {cleaned_text[:200]}")

#                 documents.append({
#                     "text": cleaned_text,
#                     "metadata": {
#                         "source": file_name,
#                         "page": page_num + 1,
#                         "section": current_section if current_section else "Unknown",
#                         "standard_type": standard_info["standard"],
#                         "standard_number": standard_info["number"],
#                         "content_type": detect_content_type(cleaned_text)
#                     }
#                 })
#             else:
#                 logger.debug(f"Skipped page {page_num + 1} in {file_name}: No extractable text (likely an image)")
    
#     return documents

# def create_text_chunks(documents: List[Dict[str, str]], 
#                       chunk_size: int = 1000,  
#                       chunk_overlap: int = 250) -> List[Dict[str, str]]:
#     """Split documents into chunks while preserving concept integrity"""
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         separators=["\n\n", "\n", ". ", ", "]
#     )
    
#     chunks = []
#     for doc in documents:
#         texts = splitter.split_text(doc["text"])
#         for j, text in enumerate(texts):
#             chunks.append({
#                 "text": text,
#                 "metadata": {
#                     **doc["metadata"],
#                     "chunk_index": j,
#                     "content_length": len(text)
#                 }
#             })
    
#     logger.debug(f"Generated {len(chunks)} chunks from documents")
#     return chunks

# def insert_into_chroma(chunks: List[Dict[str, str]], collection_name: str):
#     """Insert chunks into ChromaDB with optimized settings"""
#     chroma_client = chromadb.HttpClient(host='13.232.198.216', port=8000)
    
#     try:
#         chroma_client.delete_collection(collection_name)
#     except Exception as e:
#         logger.warning(f"Failed to delete existing collection: {e}")
    
#     collection = chroma_client.create_collection(
#         name=collection_name,
#         embedding_function=embedding_function,
#         metadata={"hnsw:space": "cosine"}
#     )
    
#     ids = [f"doc_{i}" for i in range(len(chunks))]
#     documents = [chunk["text"] for chunk in chunks]
#     metadatas = [chunk["metadata"] for chunk in chunks]  # Metadata is already cleaned
    
#     batch_size = 50
#     for i in range(0, len(documents), batch_size):
#         end_idx = min(i + batch_size, len(documents))
#         collection.add(
#             ids=ids[i:end_idx],
#             documents=documents[i:end_idx],
#             metadatas=metadatas[i:end_idx]
#         )
    
#     logger.info(f"Inserted {collection.count()} documents into ChromaDB")
#     return collection.count()

# def main():
#     pdf_dir = "pdf"
#     collection_name = "commercial_credit"
    
#     documents = get_pdf_text(pdf_dir)
#     logger.info(f"Extracted text from {len(documents)} documents")
    
#     chunks = create_text_chunks(documents)
#     logger.info(f"Created {len(chunks)} chunks")
    
#     doc_count = insert_into_chroma(chunks, collection_name)
#     logger.info(f"Successfully inserted {doc_count} documents into ChromaDB")

# if __name__ == "__main__":
#     main()


import chromadb
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
from dotenv import load_dotenv
from typing import List, Dict
import re
import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

def clean_text(text: str) -> str:
    """Clean and normalize text while preserving document structure."""
    text = re.sub(r'(?<=\n)\s*(\d+\.|\•|\-)\s*', r'\n\1 ', text)  # Preserve numbered lists and bullets
    text = re.sub(r'\n([A-Z][A-Za-z\s]+)(?=\n)', r'\n### \1', text)  # Preserve section headers
    text = re.sub(r'[^\w\s.,;:?!()\[\]{}\-–—•\n\*]', '', text)  # Remove unwanted characters
    return text.strip()

def detect_content_type(text: str) -> str:
    """Detect the type of content in the text."""
    text_lower = text.lower()
    if "definition" in text_lower or "means" in text_lower:
        return "definition"
    elif "example" in text_lower:
        return "example"
    elif any(word in text_lower for word in ["shall", "must", "required"]):
        return "requirement"
    elif "disclosure" in text_lower:
        return "disclosure"
    return "explanation"

def get_pdf_text(pdf_dir: str) -> List[Dict[str, str]]:
    """Extract text from PDFs with enhanced structure preservation."""
    documents = []
    for file_name in os.listdir(pdf_dir):
        if not file_name.endswith(".pdf"):
            continue
        file_path = os.path.join(pdf_dir, file_name)
        try:
            pdf_reader = PdfReader(file_path)
        except Exception as e:
            logger.error(f"Failed to read {file_name}: {e}")
            continue
        
        # Extract standard info from filename if applicable (not critical for your single document)
        standard_match = re.search(r'([A-Z]+)\s*(\d+)', file_name)
        standard_info = {
            "standard": standard_match.group(1) if standard_match else "",
            "number": standard_match.group(2) if standard_match else ""
        }
        
        current_section = ""
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                # Enhanced section detection using numbered sections (e.g., "6.2.1")
                section_match = re.search(r'(\d+\.\d+(\.\d+)?)\s+([A-Za-z\s]+)', text)
                if section_match:
                    current_section = f"{section_match.group(1)} {section_match.group(3).strip()}"
                elif re.search(r'\n([A-Z][A-Za-z\s]+)(?=\n)', text):
                    section_match = re.search(r'\n([A-Z][A-Za-z\s]+)(?=\n)', text)
                    current_section = section_match.group(1)
                
                cleaned_text = clean_text(text)
                logger.debug(f"Page {page_num + 1} Cleaned Text: {cleaned_text[:200]}")
                documents.append({
                    "text": cleaned_text,
                    "metadata": {
                        "source": file_name,
                        "page": page_num + 1,
                        "section": current_section if current_section else "Unknown",
                        "standard_type": standard_info["standard"],
                        "standard_number": standard_info["number"],
                        "content_type": detect_content_type(cleaned_text)
                    }
                })
            else:
                logger.debug(f"Skipped page {page_num + 1} in {file_name}: No extractable text")
    
    return documents

def create_text_chunks(documents: List[Dict[str, str]], 
                      chunk_size: int = 1000,  # Suitable for fitting full lists
                      chunk_overlap: int = 300) -> List[Dict[str, str]]:
    """Split documents into chunks while preserving concept integrity."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", "]
    )
    
    chunks = []
    for doc in documents:
        texts = splitter.split_text(doc["text"])
        for j, text in enumerate(texts):
            chunks.append({
                "text": text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": j,
                    "content_length": len(text)
                }
            })
    
    logger.debug(f"Generated {len(chunks)} chunks from documents")
    return chunks

def insert_into_chroma(chunks: List[Dict[str, str]], collection_name: str):
    """Insert chunks into ChromaDB with optimized settings."""
    chroma_client = chromadb.HttpClient(host='13.232.198.216', port=8000)
    try:
        chroma_client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception as e:
        logger.warning(f"Failed to delete existing collection: {e}")
    
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = [f"doc_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        try:
            collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            logger.debug(f"Inserted batch {i} to {end_idx}")
        except Exception as e:
            logger.error(f"Error inserting batch {i} to {end_idx}: {e}")
    
    logger.info(f"Inserted {collection.count()} documents into ChromaDB")
    return collection.count()

def main():
    pdf_dir = "pdf"
    collection_name = "commercial_credit"
    
    if not os.path.exists(pdf_dir):
        logger.error(f"Directory {pdf_dir} does not exist")
        return
    
    documents = get_pdf_text(pdf_dir)
    logger.info(f"Extracted text from {len(documents)} documents")
    
    chunks = create_text_chunks(documents)
    logger.info(f"Created {len(chunks)} chunks")
    
    doc_count = insert_into_chroma(chunks, collection_name)
    logger.info(f"Successfully inserted {doc_count} documents into ChromaDB")

if __name__ == "__main__":
    main()