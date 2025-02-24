import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline


recipes_df = pd.read_csv("cleaned_recipes.csv")


model = SentenceTransformer('all-MiniLM-L6-v2')
recipe_embeddings = model.encode(recipes_df['ingredients'].apply(lambda x: ', '.join(x)))


dimension = recipe_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(recipe_embeddings)


generator = pipeline("text2text-generation", model="t5-small")


def retrieve_recipes(user_input, top_k=5):
    user_embedding = model.encode([user_input])
    distances, indices = index.search(user_embedding, top_k)
    return recipes_df.iloc[indices[0]]


def generate_response(retrieved_recipes):
    context = " ".join(retrieved_recipes['steps'].tolist())
    response = generator(f"Summarize the recipe steps: {context}")
    return response[0]['generated_text']


def main():
    print("Welcome to the RAG Recipe Generator Bot!")
    while True:
        user_input = input("Enter ingredients or cuisine type (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        retrieved_recipes = retrieve_recipes(user_input)
        print("\nRetrieved Recipes:")
        for i, recipe in retrieved_recipes.iterrows():
            print(f"{i+1}. {recipe['name']}")

        response = generate_response(retrieved_recipes)
        print("\nGenerated Recipe Instructions:")
        print(response)

if __name__ == "__main__":
    main()