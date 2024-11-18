# Car Chatbot GenAI

![Project Banner](./image.png)

A car recommendation chatbot built using **Retrieval-Augmented Generation (RAG)** architecture. The chatbot leverages a combination of retrieval-based and generative AI techniques to provide:

- Detailed car model information.
- Car comparisons.
- Answers to user queries tailored to budget, preferences, and requirements.

This project is designed to help users make informed decisions when purchasing a car by recommending the best options based on semantic search and data-driven insights.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **AI-Powered Recommendations**: Uses a semantic search model and FAISS to retrieve the most relevant cars.
- **Customizable Queries**: Supports user-defined preferences like price range, car type (SUV, sedan, etc.), and features (e.g., sunroof).
- **High Efficiency**: Implements FAISS for fast similarity search and cosine similarity for ranking.
- **Generative Responses**: Provides detailed explanations and recommendations using Google's Generative AI API.
- **Interactive Interface**: Built with Streamlit for a conversational user experience.

---

## Tech Stack

- **Languages**: Python
- **Machine Learning**: 
  - Sentence Transformers (`all-roberta-large-v1`)
  - FAISS (Facebook AI Similarity Search)
- **Generative AI**: Google Generative AI (`gemini-1.5-flash`)
- **Frontend**: Streamlit
- **Logging**: Loguru
- **Dependencies**: `numpy`, `pandas`

---

## Installation

1. Clone the repository.

2. Set up a virtual environment (optional but recommended).

3. Install dependencies.

4. Configure the **Google Generative AI API**:
   - Obtain your API key from [Google Generative AI](https://generativeai.google.com/).
   - Replace `'API_KEY'` in the script with your actual API key.

5. Prepare the dataset:
   - Ensure your cleaned data file is available as `Cleaned_data_with_embeddings.csv` in the `data` directory.
   - If using a different dataset, modify the script accordingly.

---

## Usage

1. Run the Streamlit app.

2. Interact with the chatbot:
   - Provide your budget and preferences.
   - Receive car recommendations based on semantic similarity and FAISS indexing.
   - View detailed specifications and comparisons for the recommended cars.

---

## How It Works

### Data Preparation

- The cleaned dataset includes car specifications (price, mileage, description, and embeddings for semantic search).
- Data is preprocessed to handle missing values and convert embeddings into a numerical format.

### Recommendation Engine

1. **Semantic Search**:
   - Encodes user queries and car data using the `all-roberta-large-v1` Sentence Transformer.
   - Filters results based on user-defined budget ranges and ranks them by cosine similarity.

2. **FAISS Indexing**:
   - Uses FAISS for efficient similarity searches.
   - Measures cosine similarity for top-k recommendations.

3. **Generative Responses**:
   - Generates detailed responses by combining retrieved car data with the query context.
   - Powered by Google's Generative AI.

### Example Query

- User: "I need an SUV with a sunroof under ₹20 lakhs."
- Chatbot: Lists the top 5 recommendations with details like price, mileage, description, and semantic similarity scores.

---

## Future Enhancements

- Add multi-language support for global users.
- Integrate advanced filtering options (e.g., fuel type, brand, seating capacity).
- Expand the dataset with more car models and user reviews.
- Optimize FAISS indexing for larger datasets.
- Deploy as a web service for broader accessibility.

---

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch (`feature/new-feature`).
3. Commit your changes.
4. Open a Pull Request.

Please ensure your contributions align with the project's goals and follow the coding style outlined in the repository.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Google Generative AI for its robust generative capabilities.
- Facebook AI for the FAISS library.
- The open-source community for the `SentenceTransformers` library.
- [Streamlit](https://streamlit.io/) for making app deployment straightforward.

---
