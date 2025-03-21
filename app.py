import streamlit as st
import pytesseract
from PIL import Image
import PyPDF2
import docx

import requests

from pinecone import Pinecone





api_key = "pcsk_hQtiV_LB7cGEvwms8YM7VyeVfNRQMRUTt9YfFKFzxM4usi2tre4HdyFsAxzLLZkbPXjvX"
pc = Pinecone(api_key=api_key)
index = pc.Index("sanjeev")


def store_data(chunk, embedding):
    try:
        # Create unique ID for the chunk
        import uuid
        chunk_id = str(uuid.uuid4())
        
        # Store single vector
        index.upsert(
            vectors=[{
                "id": chunk_id,
                "values": embedding,
                "metadata": {"chunk": chunk}
            }],
            namespace="tenant1"
        )
        return chunk_id
        
    except Exception as e:
        print(f"Error storing data: {str(e)}")
        return None

def retrieve_data(prompt):
    try:
        # Query the index
        query_results = index.query(
            namespace="tenant1",
            vector=prompt,
            top_k=3,
            include_metadata=True  # Make sure to include metadata
        )
        
        # Extract chunks from metadata
        retrieved_docs = []
        if "matches" in query_results:
            for match in query_results["matches"]:
                if "metadata" in match and "chunk" in match["metadata"]:
                    retrieved_docs.append(match["metadata"]["chunk"])
        
        return retrieved_docs
    
    except Exception as e:
        print(f"Error retrieving data: {str(e)}")
        return []

# print(query_results)

Jina_api = "jina_d1f20ceaa138457c8f8fe46db436665b3xRYUXn_hqA5lUuHnlDF9xH5kpGZ"
embedding_api=Jina_api   

def embeddingprocess(text):
    headers = {
        "Authorization": f"Bearer {embedding_api}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "jina-clip-v2",
        "dimensions": 1024,
        "normalized": True,
        "embedding_type": "float",
        "input": [
            {
                "text": text
            }
        ]
    }
    
    try:
        # Added timeout parameters: 30 seconds for connection, 90 seconds for read
        response = requests.post(
            "https://api.jina.ai/v1/embeddings", 
            json=payload, 
            headers=headers,
            timeout=(30, 90)  # (connection timeout, read timeout)
        )
        
        if response.status_code == 200:
            embedding = response.json()['data'][0]['embedding']
            if isinstance(embedding, list):
                return embedding
            return None
        else:
            print(f"API Error: Status {response.status_code}, Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("Request timed out while connecting to Jina AI API")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error making request to Jina AI API: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

from langchain_groq import ChatGroq


GROQ_API_KEY = "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"
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)
def llm_response(input_text):
    print(f"input_text", input_text)
    prompt = f"""  
    role : system context:You are an AI assistant that answers based on retrieved documents
    role : User context: {input_text}
    
      """
    response = llm.invoke(prompt)
    
    final_response = response.content 
  
    return final_response


def chunk_text(text, chunk_size=100, overlap=10):
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []
    embeddings = []
    
    print(f"\n=== Chunking Details ===")
    print(f"Total words: {len(words)}")
    print(f"Chunk size: {chunk_size}")
    print(f"Overlap size: {overlap}")
    
    if len(words) <= chunk_size:
        chunk = text
        print(f"\nSingle chunk created (text shorter than chunk size)")
        print(f"Chunk length: {len(chunk)} characters")
        embedding = embeddingprocess(chunk)
        if embedding:
            store_data(chunk, embedding)
            print(f"Embedding created with dimension: {len(embedding)}")
            return [chunk], [embedding]
        return [], []
        
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        print(f"\nChunk {len(chunks) + 1}:")
        print(f"Start index: {i}")
        print(f"End index: {i + chunk_size}")
        print(f"Chunk length: {len(chunk)} characters")
        embedding = embeddingprocess(chunk)
        if embedding:
            chunks.append(chunk)
            embeddings.append(embedding)
            store_data(chunk, embedding)
            print(f"Embedding created with dimension: {len(embedding)}")
    
    print(f"\nTotal chunks created: {len(chunks)}")
    print("========================\n")
    return chunks, embeddings

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []

def extract_text_from_pdf(pdf_file):
    try:
        print("Starting PDF text extraction...")
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        print(f"Number of pages in PDF: {len(pdf_reader.pages)}")
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text
            print(f"Extracted text from page {page_num + 1}: {len(page_text)} characters")
        print("PDF text extraction completed.")
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {str(e)}")
        st.error(f"Error extracting PDF text: {str(e)}")
        return None

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting DOCX text: {str(e)}")
        return None

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.sidebar.error(f"Error extracting text from image: {str(e)}")
        return None

def main():
    st.title("Document Chat System")

    # Sidebar - File Upload
    with st.sidebar:
        st.header("File Upload")
        uploaded_file = st.file_uploader("Upload a file", 
                                       type=['txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            st.info(f"File: {uploaded_file.name}")
            
            try:
                if uploaded_file.type in ["image/jpeg", "image/png"]:
                    text = extract_text_from_image(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    text = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_text_from_docx(uploaded_file)
                
                if text:
                    st.session_state.extracted_text = text
                    # Create chunks and get embeddings
                    chunks, embeddings = chunk_text(text)
                    st.session_state.text_chunks = chunks
                    
                    with st.expander("Extracted Text"):
                        st.text(text)
                    with st.expander("Embeddings"):
                        st.write(embeddings)
                    with st.expander("Text Chunks and Embeddings"):
                        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                            st.write(f"Chunk {i+1}:")
                            st.text(chunk)
                            st.write(f"Embedding shape: {len(embedding)}")
                            st.divider()
                    
                    st.metric("Word Count", len(text.split()))
                    st.metric("Character Count", len(text))
                    st.metric("Number of Chunks", len(st.session_state.text_chunks))
                    
                    st.download_button(
                        label="Download extracted text",
                        data=text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                    
            except Exception as e:
                st.sidebar.error(f"Error processing file: {str(e)}")

    # Main area - Chat Interface
    st.header("Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the document"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Process query and get response
        embed_search = embeddingprocess(prompt)
        if embed_search:
            retrieved = retrieve_data(embed_search)
            if retrieved:
                response = f"user query{prompt} \n\n"
                for i, doc in enumerate(retrieved, 1):
                    response += f"{i}. {doc}\n"
                response = llm_response(response)
            else:
                response = "No relevant information found in the document."
        else:
            response = "Sorry, I couldn't process your query."
            
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()

  