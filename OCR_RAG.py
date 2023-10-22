import os
import streamlit as st
from google.cloud import vision
from google.oauth2.service_account import Credentials
import openai
import pinecone

st.write("Streamlit is running!")  # Debug 1: Check basic Streamlit functionality

# Google Cloud Vision API setup
creds = Credentials.from_service_account_file("location.json")
client = vision.ImageAnnotatorClient(credentials=creds)

def ocr_with_vision_api(image_file) -> str:
    """Use Google Cloud Vision API to extract text from an image."""
    content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        return texts[0].description
    else:
        return "No text detected"

# Pinecone setup
os.environ['PINECONE_ENVIRONMENT'] = 'gcp-starter'
os.environ['PINECONE_API_KEY'] = 'PINECONE_API_KEY'
api_key = os.getenv("PINECONE_API_KEY") or "PINECONE_API_KEY"
env = os.getenv("PINECONE_ENVIRONMENT") or "PINECONE_ENVIRONMENT"

# Debug 3: Check Pinecone Initialization
try:
    pinecone.init(api_key=api_key, environment=env)
    st.write("Pinecone initialized successfully.")
except Exception as e:
    st.write("Error initializing Pinecone:")
    st.write(e)

index_name = 'houseofthesky'
index = pinecone.GRPCIndex(index_name)


# OpenAI setup
if openai.api_key is None:
    openai.api_key = "openai_api_key"

def generate_augmented_query(query, embed_model, k=5):
    query_embedding = openai.Embedding.create(input=[query], engine=embed_model)
    xq = query_embedding['data'][0]['embedding']
    res = index.query(xq, top_k=k, include_metadata=True)
    contexts = [item['metadata'].get('text', 'Default Value') for item in res['matches']]
    return "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query

def rag_response(query):
    embed_model = 'text-embedding-ada-002'
    primer = """
      You are a Q&A bot. A highly intelligent system that answers
      user questions based on the information provided by the user above
      each question. If the information can not be found in the information
      provided by the user you truthfully say "I don't know".
      """
    model = "gpt-4"
    comp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": generate_augmented_query(query, embed_model)}
        ])
    if comp and comp.get('choices'):
        st.write(comp['choices'][0]['message']['content'])  # Debug 4: Check OpenAI Functionality
    else:
        st.write("No response received from OpenAI.")



import streamlit as st
from google.cloud import vision
from google.oauth2.service_account import Credentials
import os
import getpass
import openai
import pinecone
from IPython.display import Markdown

# Google Cloud Vision Initialization
creds = Credentials.from_service_account_file(r"C:\Users\Laur\python_projects\Training\week 8\GoogleCloudAI\cloudaiocr-b1c85d046e41.json")
client = vision.ImageAnnotatorClient(credentials=creds)

# Pinecone Initialization
os.environ['PINECONE_ENVIRONMENT'] = 'gcp-starter'
os.environ['PINECONE_API_KEY'] = '0389c178-d738-418b-a642-16cb05f2ce0a'
api_key = os.getenv("PINECONE_API_KEY") or "PINECONE_API_KEY"
env = os.getenv("PINECONE_ENVIRONMENT") or "PINECONE_ENVIRONMENT"
pinecone.init(api_key=api_key, environment=env)
index_name = 'houseofthesky'
index = pinecone.GRPCIndex(index_name)

if openai.api_key is None:
    openai.api_key = "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"

def ocr_with_vision_api(image_file) -> str:
    """Use Google Cloud Vision API to extract text from an image."""
    content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        return texts[0].description
    else:
        return "No text detected"

def generate_augmented_query(query, embed_model, k=5):
  query_embedding = openai.Embedding.create(input=[query], engine=embed_model)
  xq = query_embedding['data'][0]['embedding']
  res = index.query(xq, top_k=k, include_metadata=True)
  contexts = [item['metadata']['data'] for item in res['matches']]
  return "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query

def rag_response(query):
  embed_model = 'text-embedding-ada-002'
  primer = """
  You are an expert writer, a narator, a professional story teller.You will only provide reponses from the book 'HOUSE of SKY and BREATH' a crescent city novel by SARAH J. MAAS. If the information can not be found in the book you truthfully say only :'I don't know'! nothing else!
  """
  model = "gpt-4"
  return llm_answer(model, primer, generate_augmented_query(query, embed_model))

def llm_answer(llmmodel, primer, augmented_query):
  comp = openai.ChatCompletion.create(
    model=llmmodel,
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ])
  return comp['choices'][0]['message']['content']

# Streamlit Application
st.title("Handwritten Text Recognition and Information Retrieval")

uploaded_file = st.file_uploader("Upload an image containing handwritten text", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    with st.spinner('Processing...'):
        extracted_text = ocr_with_vision_api(uploaded_file)
        edited_text = st.text_area("Edit Extracted Text:", value=extracted_text, height=200)
        
        if st.button("Generate Response"):
            with st.spinner('Generating response...'):
                response = rag_response(edited_text)
                st.write("Response based on the text:")
                st.write(response)
else:
    st.write("Please upload an image to start the process.")
