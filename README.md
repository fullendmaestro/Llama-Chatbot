# Llama-2 Enhanced Chatbot with Sentiment Analysis

![Chat Interface](./chat_interface.png)

This project implements an interactive chatbot using the **Llama-2 model** for text generation, enhanced with **sentiment analysis** for improved user interaction. The chatbot can handle various types of user queries, perform semantic searches in different datasets, and adjust its responses based on the emotional tone of the user’s input.

## Features

- **Llama-2 Integration**: Uses the Llama-2 model from Hugging Face for high-quality, conversational text generation.
- **Sentiment Analysis**: Detects the user's sentiment (positive, neutral, or negative) and adapts responses accordingly.
- **Semantic Search**: Uses Pinecone for fast, vector-based searches across datasets (QA, products, troubleshooting).
- **Contextual Responses**: Incorporates context from search results and chat history for more accurate and personalized responses.

## Project Overview

1. **Sentiment Analysis**  
   The chatbot detects whether the user's message is positive, neutral, or negative using a pre-trained sentiment analysis model from Hugging Face (`distilbert-base-uncased-finetuned-sst-2-english`).

2. **Llama-2 for Response Generation**  
   Uses the `NousResearch/Llama-2-7b-chat-hf` model from Hugging Face, capable of generating dynamic and engaging responses based on user queries.

3. **Pinecone for Semantic Search**  
   The project integrates Pinecone for fast and efficient semantic search across multiple datasets (e.g., Q&A, product descriptions, troubleshooting guides). This enhances the bot’s ability to provide accurate, contextually relevant responses.

4. **Adapted Responses**  
   The chatbot tailors its tone and style of responses based on the sentiment of the user’s query:
   - **Negative sentiment**: The bot is empathetic and supportive.
   - **Positive sentiment**: The bot is upbeat and conversational.
   - **Neutral sentiment**: The bot remains formal and informative.

## Project Files

- `products.csv`: Contains product descriptions and metadata.
- `qa_dataset.csv`: Contains a Q&A dataset for answering common questions.
- `troubleshooting.csv`: Contains troubleshooting steps for resolving issues.

## Installation

First make sure you copied all the required files into your colab notebook. And also make sure to set all the required secrets in your colab notebook.

### Requirements

- Python 3.x
- Google Colab or a local environment with GPU access
- Hugging Face Transformers
- Pinecone API
- Gradio for the web interface

### Python Libraries

To install the necessary dependencies, run:

```bash
pip install transformers pandas torch sentence-transformers pinecone-client gradio

```

## Hugging Face and Pinecone API Keys

Before running the code, set up your API keys for Hugging Face and Pinecone. You can store them in Google Colab using the `userdata` secret method or in environment variables. For example using colab secret.

```bash
from google.colab import userdata
pinecone_api_key = userdata.get('PINECONE_API_KEY')
huggingface_api_key = userdata.get('HUGGINGFACEHUB_API_TOKEN')
openai_api_key = userdata.get('OPENAI_API_KEY')
pinecone_region = userdata.get('PINECONE_ENV')


```

## Key Functions

### 1. **`embed_and_store_documents`**

**Purpose**: Embeds text data using a sentence-transformer model and stores the resulting vectors in the Pinecone vector database.

- **Arguments**:
  - `df`: A pandas DataFrame containing the dataset.
  - `text_column`: The name of the column(s) in the DataFrame that contains the text to embed.
  - `namespace`: The namespace to store the embeddings in Pinecone.
  - `metadata_columns`: List of additional metadata columns to store with the embeddings.
- **Process**:
  - Iterates over each row in the DataFrame.
  - Embeds the text using the `sentence-transformers/multi-qa-mpnet-base-cos-v1` model.
  - Uploads embeddings and associated metadata to Pinecone.

### 2. **`index_datasets`**

**Purpose**: Loads the QA, product, and troubleshooting datasets, then embeds and stores them in Pinecone using the `embed_and_store_documents` function.

- **Process**:
  - Loads the datasets (`qa_dataset.csv`, `products.csv`, `troubleshooting.csv`).
  - Embeds the "Answer" (QA), "Description" (Products), and "Issue + Steps" (Troubleshooting) columns.
  - Stores the embeddings in their respective namespaces ("qa", "products", "troubleshooting") in Pinecone.

### 3. **`semantic_search`**

**Purpose**: Performs a semantic search for a user query by generating a vector embedding and querying the Pinecone index for the most similar documents.

- **Arguments**:
  - `query`: The user's input query.
  - `namespace`: The specific namespace (e.g., "qa", "products", "troubleshooting") to search in.
- **Returns**: The top search results with their associated metadata.
- **Process**:
  - Encodes the query using the sentence-transformer model.
  - Queries Pinecone using the query embedding and returns the top-k results.

### 4. **`generate_with_context`**

**Purpose**: Generates a chatbot response based on the user question, semantic search results, and conversation history.

- **Arguments**:
  - `question`: The user’s question.
  - `search_results`: Results from the semantic search.
  - `history`: A list of previous user and chatbot interactions.
- **Process**:
  - Formats the history and search results into a context prompt.
  - Feeds the full prompt into the Llama-2 model to generate a response.
- **Returns**: The chatbot's response text, trimmed to exclude the original prompt.

### 5. **`answer_question`**

**Purpose**: Combines the chatbot's main logic: semantic search, response generation, and sentiment adjustment.

- **Arguments**:
  - `question`: The user’s question.
  - `history`: Conversation history up to that point.
- **Process**:
  - Executes the semantic search across all relevant namespaces (QA, products, troubleshooting).
  - Generates a response using `generate_with_context`.
  - Adjusts the response based on sentiment analysis of the user input.
- **Returns**: The chatbot's adjusted response.

### 6. **`gradio_chat_interface`**

**Purpose**: Provides a Gradio-based chat interface that takes user input and passes it through the chatbot's logic.

- **Arguments**:
  - `question`: User’s question from the Gradio input.
  - `history`: List of previous user-chatbot interactions.
- **Returns**: The chatbot's response to display in the Gradio chat interface.
- **Process**:
  - Displays input and interaction history for debugging.
  - Calls `answer_question` to generate the response.

## Secret variables for submision

`HUGGINGFACEHUB_API_TOKEN=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`
`OPENAI_API_KEY=sk-2Wte_TRhsDi-pt1ZS8G8TCzNGfrp-ycgVCY4Xu3_0iT3BlbkFJO9y7jtYuJIo-yL2rt8bE06D4kOiMzQIl1g3r4WQuwA`
`PINECONE_API_KEY=b6fa6a70-9ea4-40fc-b393-d53318b566fc`
`PINECONE_ENV=us-east-1`

# Llama-2 Enhanced Chatbot with Sentiment Analysis

![Chat Interface](./chat_interface.png)

This project implements an interactive chatbot using the **Llama-2 model** for text generation, enhanced with **sentiment analysis** for improved user interaction. The chatbot can handle various types of user queries, perform semantic searches in different datasets, and adjust its responses based on the emotional tone of the user’s input.

## Features

- **Llama-2 Integration**: Uses the Llama-2 model from Hugging Face for high-quality, conversational text generation.
- **Sentiment Analysis**: Detects the user's sentiment (positive, neutral, or negative) and adapts responses accordingly.
- **Semantic Search**: Uses Pinecone for fast, vector-based searches across datasets (QA, products, troubleshooting).
- **Contextual Responses**: Incorporates context from search results and chat history for more accurate and personalized responses.

## Project Overview

1. **Sentiment Analysis**  
   The chatbot detects whether the user's message is positive, neutral, or negative using a pre-trained sentiment analysis model from Hugging Face (`distilbert-base-uncased-finetuned-sst-2-english`).

2. **Llama-2 for Response Generation**  
   Uses the `NousResearch/Llama-2-7b-chat-hf` model from Hugging Face, capable of generating dynamic and engaging responses based on user queries.

3. **Pinecone for Semantic Search**  
   The project integrates Pinecone for fast and efficient semantic search across multiple datasets (e.g., Q&A, product descriptions, troubleshooting guides). This enhances the bot’s ability to provide accurate, contextually relevant responses.

4. **Adapted Responses**  
   The chatbot tailors its tone and style of responses based on the sentiment of the user’s query:
   - **Negative sentiment**: The bot is empathetic and supportive.
   - **Positive sentiment**: The bot is upbeat and conversational.
   - **Neutral sentiment**: The bot remains formal and informative.

## Project Files

- `products.csv`: Contains product descriptions and metadata.
- `qa_dataset.csv`: Contains a Q&A dataset for answering common questions.
- `troubleshooting.csv`: Contains troubleshooting steps for resolving issues.

## Installation

First make sure you copied all the required files into your colab notebook. And also make sure to set all the required secrets in your colab notebook.

### Requirements

- Python 3.x
- Google Colab or a local environment with GPU access
- Hugging Face Transformers
- Pinecone API
- Gradio for the web interface

### Python Libraries

To install the necessary dependencies, run:

```bash
pip install transformers pandas torch sentence-transformers pinecone-client gradio

```

## Hugging Face and Pinecone API Keys

Before running the code, set up your API keys for Hugging Face and Pinecone. You can store them in Google Colab using the `userdata` secret method or in environment variables. For example using colab secret.

```bash
from google.colab import userdata
pinecone_api_key = userdata.get('PINECONE_API_KEY')
huggingface_api_key = userdata.get('HUGGINGFACEHUB_API_TOKEN')
openai_api_key = userdata.get('OPENAI_API_KEY')
pinecone_region = userdata.get('PINECONE_ENV')


```

## Key Functions

### 1. **`embed_and_store_documents`**

**Purpose**: Embeds text data using a sentence-transformer model and stores the resulting vectors in the Pinecone vector database.

- **Arguments**:
  - `df`: A pandas DataFrame containing the dataset.
  - `text_column`: The name of the column(s) in the DataFrame that contains the text to embed.
  - `namespace`: The namespace to store the embeddings in Pinecone.
  - `metadata_columns`: List of additional metadata columns to store with the embeddings.
- **Process**:
  - Iterates over each row in the DataFrame.
  - Embeds the text using the `sentence-transformers/multi-qa-mpnet-base-cos-v1` model.
  - Uploads embeddings and associated metadata to Pinecone.

### 2. **`index_datasets`**

**Purpose**: Loads the QA, product, and troubleshooting datasets, then embeds and stores them in Pinecone using the `embed_and_store_documents` function.

- **Process**:
  - Loads the datasets (`qa_dataset.csv`, `products.csv`, `troubleshooting.csv`).
  - Embeds the "Answer" (QA), "Description" (Products), and "Issue + Steps" (Troubleshooting) columns.
  - Stores the embeddings in their respective namespaces ("qa", "products", "troubleshooting") in Pinecone.

### 3. **`semantic_search`**

**Purpose**: Performs a semantic search for a user query by generating a vector embedding and querying the Pinecone index for the most similar documents.

- **Arguments**:
  - `query`: The user's input query.
  - `namespace`: The specific namespace (e.g., "qa", "products", "troubleshooting") to search in.
- **Returns**: The top search results with their associated metadata.
- **Process**:
  - Encodes the query using the sentence-transformer model.
  - Queries Pinecone using the query embedding and returns the top-k results.

### 4. **`generate_with_context`**

**Purpose**: Generates a chatbot response based on the user question, semantic search results, and conversation history.

- **Arguments**:
  - `question`: The user’s question.
  - `search_results`: Results from the semantic search.
  - `history`: A list of previous user and chatbot interactions.
- **Process**:
  - Formats the history and search results into a context prompt.
  - Feeds the full prompt into the Llama-2 model to generate a response.
- **Returns**: The chatbot's response text, trimmed to exclude the original prompt.

### 5. **`answer_question`**

**Purpose**: Combines the chatbot's main logic: semantic search, response generation, and sentiment adjustment.

- **Arguments**:
  - `question`: The user’s question.
  - `history`: Conversation history up to that point.
- **Process**:
  - Executes the semantic search across all relevant namespaces (QA, products, troubleshooting).
  - Generates a response using `generate_with_context`.
  - Adjusts the response based on sentiment analysis of the user input.
- **Returns**: The chatbot's adjusted response.

### 6. **`gradio_chat_interface`**

**Purpose**: Provides a Gradio-based chat interface that takes user input and passes it through the chatbot's logic.

- **Arguments**:
  - `question`: User’s question from the Gradio input.
  - `history`: List of previous user-chatbot interactions.
- **Returns**: The chatbot's response to display in the Gradio chat interface.
- **Process**:
  - Displays input and interaction history for debugging.
  - Calls `answer_question` to generate the response.

# Submission files

## Secret variables for submision

`HUGGINGFACEHUB_API_TOKEN=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`
`OPENAI_API_KEY=sk-2Wte_TRhsDi-pt1ZS8G8TCzNGfrp-ycgVCY4Xu3_0iT3BlbkFJO9y7jtYuJIo-yL2rt8bE06D4kOiMzQIl1g3r4WQuwA`
`PINECONE_API_KEY=b6fa6a70-9ea4-40fc-b393-d53318b566fc`
`PINECONE_ENV=us-east-1`

## Google colab link

https://colab.research.google.com/drive/1ZfX_VMZ4t8fpRZ-3XXW71tMKwIN9nnJa?usp=sharing
