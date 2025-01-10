import os
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain_groq import ChatGroq
from faiss import IndexFlatL2

# Initialize FAISS vector store
def init_faiss():
    # Using HuggingFace embeddings for text representation
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists("faiss_chat_index"):
        # Load the FAISS index if it exists
        vectorstore = FAISS.load_local("faiss_chat_index", embeddings, allow_dangerous_deserialization=True)
    else:
        # Create a new FAISS index if it doesn't exist
        sample_embedding = embeddings.embed_query("sample")  # Embed a sample query
        index = IndexFlatL2(len(sample_embedding))  # FAISS index with embedding dimensions
        docstore = InMemoryDocstore({})  # In-memory docstore
        index_to_docstore_id = {}  # Empty mapping

        # Initialize FAISS with the embedding function
        vectorstore = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

        # Save the FAISS index locally
        vectorstore.save_local("faiss_chat_index")

    return vectorstore

# Save user message to FAISS
def save_to_faiss(vectorstore, user_message, metadata):
    vectorstore.add_texts([user_message], metadatas=[metadata])  # Add new texts to vectorstore
    vectorstore.save_local("faiss_chat_index")  # Save after updating

# Analyze sentiment of user message
def analyze_sentiment(user_message):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(user_message)
    if sentiment_scores['pos'] > sentiment_scores['neg']:
        return 'Positive'
    elif sentiment_scores['neg'] > sentiment_scores['pos']:
        return 'Negative'
    else:
        return 'Neutral'

def generate_response(user_message, sentiment, groq_model, vectorstore):
    tone = {
        'Positive': "happy and supportive",
        'Negative': "empathetic and understanding",
        'Neutral': "neutral and conversational"
    }[sentiment]

    # Retrieve relevant context from the vector store
    context = vectorstore.similarity_search(user_message, k=2)
    context_text = " ".join([c.metadata.get('user_message', '') for c in context])

    # Create a structured prompt for Groq
    prompt = f"""
    Act as an emotionally intelligent friend. Respond naturally, mimicking a human who reflects emotions.
    Your tone should be {tone} based on the sentiment. Context is provided for a better response.

    Context: {context_text}

    User: {user_message}
    AI:
    """

    # Updated method to call Groq model (use .invoke())
    response = groq_model.invoke([{"role": "user", "content": prompt}])  # Proper format for Groq model

    # Access the response content correctly
    return response.content.strip()  # Use the .content attribute of the AIMessage object


# Initialize components
vectorstore = init_faiss()
api_key = "YOUR_API_KEY_FOR_GROQ" 
groq_model = ChatGroq(temperature=0.3, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")



# Chat loop
print("Chat with the AI! Type 'exit' or 'quit' to end the session.")
while True:
    user_message = input("You: ").strip()
    if user_message.lower() in ['quit', 'exit']:
        print("Exiting chat. Goodbye!")
        break

    # Process the user's message
    sentiment = analyze_sentiment(user_message)
    metadata = {"user_message": user_message, "sentiment": sentiment, "timestamp": str(datetime.now())}
    save_to_faiss(vectorstore, user_message, metadata)

    # Generate AI response
    response = generate_response(user_message, sentiment, groq_model, vectorstore)
    print(f"AI: {response}")
