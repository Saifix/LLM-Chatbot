import transformers
from transformers import pipeline

# Load pre-trained LLM model (replace with your chosen model and library)
model_name = "facebook/bart-base"  # Adjust as needed
nlp = pipeline("question-answering", model=model_name)

# Expanded knowledge base with various categories and examples

def answer_query(question):
  
  knowledge_base = {
    "product_features": {
        "phone_x": {"camera": "12MP", "battery": "4000mAh"},
        "phone_y": {"camera": "20MP", "battery": "5000mAh"},
        "phones": {"camera": "varies", "battery": "varies"},
        "laptops": {"processor": "i7", "RAM": "16GB"},
        "cars": {"models": ["sedan", "SUV", "coupe"], "engines": ["gasoline", "electric"]}
    },
    "policies": {
        "return": "You can return the product within 30 days of purchase.",
        "return policy": "You can return the product within 30 days of purchase."
    },
    "general_knowledge": {
        "capitals": {
            "France": "Paris",
            "USA": "Washington D.C.",
            "UK": "London"
        },
        "history": {
            "WWII": "World War II, a global war that lasted from 1939 to 1945."
        },
        "science": {
            "gravity": "The force by which a planet or other massive object attracts objects to it."
        }
    }
}

  """
  Processes user query, performs basic cleaning, and retrieves relevant answer from knowledge base.
  """
  processed_question = question.lower().strip()
  # Consider adding more specific cleaning steps here if needed (e.g., removing special characters)

  category = None
  for word in processed_question.split():
    if word in knowledge_base:
      category = word
      break

  if category:
    answer = nlp(question=question, context=knowledge_base[category])["answer"]
    return answer
  else:
    return "I'm still learning and might not know everything yet. Can you rephrase your question or try asking something different?"

# User interaction loop
while True:
  user_input = input("User: ")
  response = answer_query(user_input)
  print("Chatbot:", response)
