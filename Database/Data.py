from pinecone import Pinecone
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
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