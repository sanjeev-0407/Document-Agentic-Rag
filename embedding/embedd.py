import requests
import os
from dotenv import load_dotenv
load_dotenv()
Jina_api = os.getenv("JINA_API_KEY")
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