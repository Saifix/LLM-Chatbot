import transformers
from transformers import pipeline

# Load pre-trained LLM model (replace with your chosen model and library)
model_name = "facebook/bart-base"  # Adjust as needed
nlp = pipeline("question-answering", model=model_name)

# Knowledge base with synonyms and broader concepts
knowledge_base = {
    "product_features": {
        "phone_x": {"camera": "12MP", "battery": "4000mAh"},
        "phone_y": {"camera": "20MP", "battery": "5000mAh"},
        "phones": {"camera": "varies", "battery": "varies"}  # Broader concept for "product"
    },
    "policies": {
        "return": "You can return the product within 30 days of purchase.",
        "return policy": "You can return the product within 30 days of purchase."  # Synonym for "return"
    }
}

import nltk  # Import NLTK library for basic text processing

def answer_query(question):
  """
  Processes user query, performs basic cleaning, and retrieves relevant answer from knowledge base.
  """
  # Preprocess the question (lowercase, remove punctuation, tokenize)
  processed_question = nltk.word_tokenize(question.lower().strip())

  # Identify the relevant knowledge base category based on keywords and synonyms
  category = None
  for word in processed_question:
    if word in knowledge_base:
      category = word
      break  # Stop searching after finding the first matching category

  # If a category is found, use the LLM for information retrieval within that category
  if category:
    # Use the LLM for question answering within the chosen category data
    answer = nlp(question=question, context=knowledge_base[category])["answer"]
    return answer
  else:
    return "I'm still learning and might not understand everything yet. Can you rephrase your question?"

# User interaction loop (replace with actual user input and response display)
while True:
  user_input = input("User: ")
  response = answer_query(user_input)
  print("Chatbot:", response)
