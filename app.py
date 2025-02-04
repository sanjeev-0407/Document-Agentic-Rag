import streamlit as st
import pytesseract
from PIL import Image
import PyPDF2
import docx
from embedding.embedd import embeddingprocess
from Database.Data import store_data, retrieve_data
from LLM.llm import llm_response

def chunk_text(text, chunk_size=50, overlap=10):
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []
    embeddings = []
    
    if len(words) <= chunk_size:
        chunk = text
        embedding = embeddingprocess(chunk)
        if embedding:
            store_data(chunk, embedding)
            return [chunk], [embedding]
        return [], []
        
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        embedding = embeddingprocess(chunk)
        if embedding:
            chunks.append(chunk)
            embeddings.append(embedding)
            store_data(chunk, embedding)
    
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
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
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