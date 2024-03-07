import transformers
from transformers import pipeline

# Load pre-trained LLM model (replace with your chosen model and library)
model_name = "facebook/bart-base"  # Adjust as needed
nlp = pipeline("question-answering", model=model_name)

# Knowledge base with dictionary structure aligned with categories
knowledge_base = {
    "product_features": {
        "phone_x": {"camera": "12MP", "battery": "4000mAh"},
        "phone_y": {"camera": "20MP", "battery": "5000mAh"},
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

def answer_query(question):
  """
  Processes user query, performs basic cleaning, and retrieves relevant answer from knowledge base.
  """
  processed_question = question.lower().strip()

  # Ensure category is a valid key in the knowledge base
  category = None
  if processed_question in knowledge_base:
    category = processed_question  # Use the entire question as the category if it matches a top-level key

  if category:
    # Provide the entire knowledge base category as context (adjust based on your LLM model's requirements)
    answer = nlp(question=question, context=knowledge_base[category])["answer"]
    return answer
  else:
    return "I'm still learning and might not know everything yet. Can you rephrase your question or try asking something different?"

# User interaction loop
while True:
  user_input = input("User: ")
  response = answer_query(user_input)
  print("Chatbot:", response)
