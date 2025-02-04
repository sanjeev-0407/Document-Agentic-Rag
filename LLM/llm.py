from langchain_groq import ChatGroq
from dotenv import load_dotenv  
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
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

