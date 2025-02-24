RAG Recipe Generator Bot

The RAG Recipe Generator Bot is a Python-based application that leverages Retrieval-Augmented Generation (RAG) to help users discover and generate recipes based on their input ingredients or cuisine preferences. The bot combines a FAISS-based retrieval system for efficient recipe search with a Hugging Face Transformers model for generating step-by-step cooking instructions.

Key Features
Semantic Recipe Retrieval: Uses Sentence-BERT embeddings and FAISS to find the most relevant recipes from a large dataset (e.g., Food.com).

Text Generation: Generates coherent recipe instructions using a fine-tuned T5 model.

Customizable Input: Allows users to input ingredients or cuisine types and receive tailored recipe suggestions.

Efficient and Scalable: Designed for fast retrieval and generation, even with large datasets.

Technologies Used
Python: Core programming language.

FAISS: For efficient similarity search and retrieval.

Sentence-BERT: For generating semantic embeddings of recipes.

Hugging Face Transformers: For text generation using pre-trained models like T5.

Pandas: For dataset preprocessing and manipulation.

I used this dataset on Kaggle: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
