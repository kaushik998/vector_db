# VectorDB — Vector Database from Scratch in Java

A lightweight vector database built from scratch using Java that supports similarity search, vector indexing, and semantic document retrieval. The project provides a web-based interface to interact with the vector database and explore nearest-neighbor search.

## 🚀 Features

- **Multiple Search Algorithms:** Supports HNSW (Hierarchical Navigable Small World), KD-Tree, and Brute-Force search.
- **Distance Metrics:** Supports Cosine Similarity, Euclidean Distance, and Manhattan Distance.
- **Vector Search:** Finds the most relevant vectors based on similarity to a query vector.
- **Semantic Search:** Uses text embeddings to represent text as numerical vectors and retrieve semantically relevant content.
- **RAG Pipeline:** Supports Retrieval-Augmented Generation (RAG) by retrieving relevant information to provide context for an LLM.
- **Ollama Integration:** Can use Ollama for generating text embeddings and connecting to a local language model, if configured.
- **Web Interface:** Provides an HTML-based frontend for interacting with the system.
- **Java HTTP Server:** Handles incoming requests and serves the backend functionality.

## 🛠️ Technologies Used

- Java
- HTML, CSS, JavaScript
- Vector Search Algorithms: HNSW, KD-Tree, Brute Force
- Similarity Metrics: Cosine, Euclidean, Manhattan
- Ollama (optional, depending on configuration)
- Retrieval-Augmented Generation (RAG)

## 🏗️ Architecture

The project consists of two main parts:

### 1. Query

- Accepts a search query from the user.
- Converts the query into a vector embedding when embedding support is configured.
- Searches the vector database using the selected search algorithm.
- Ranks results according to the selected similarity metric.
- Returns the most relevant matching documents or vectors.
- If RAG is enabled, retrieved information can be supplied to an LLM to generate a contextual response.

### 2. Vector Search Engine

- Stores vectors along with their associated information.
- Uses indexing and similarity search algorithms to find nearest neighbors.
- Supports exact search through Brute Force and indexed search through KD-Tree or HNSW.
- Compares vectors using configurable distance metrics.

## 🔍 Search Algorithms

| Algorithm | Description |
|---|---|
| Brute Force | Compares the query vector with every stored vector. |
| KD-Tree | Organizes vectors spatially to accelerate suitable nearest-neighbor queries. |
| HNSW | Uses a hierarchical graph for efficient approximate nearest-neighbor (ANN) search. |

## 📐 Distance Metrics

- **Cosine Similarity:** Measures the directional similarity between vectors.
- **Euclidean Distance:** Measures straight-line distance between vectors.
- **Manhattan Distance:** Measures the sum of absolute differences across dimensions.

## ⚙️ Requirements

- Java Development Kit (JDK)
- Visual Studio Code or another Java-compatible IDE
- A modern web browser
- Ollama and a compatible model, if required by your configuration

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/VectorDB.git
cd VectorDB
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Compile the Java program

If `main.java` contains the main class and all required Java source code is in that file:

```bash
javac main.java
```

### 3. Start the backend

```bash
java main
```

The server should start on the port configured in the application. In the current development setup, the terminal displays a local server address such as:

```text
http://localhost:8080
```

Use the actual address printed by your program.

### 4. Open the frontend

If the frontend is served separately, open `index.html` using a local development server, such as the VS Code Live Server extension.

Make sure the frontend API URL matches the backend address and port.

## 🧠 What Problem Does This Project Solve?

Traditional keyword-based search can miss relevant information when the user's wording differs from the wording in a document.

Vector databases represent text or other data as numerical embeddings. By comparing these vectors, the system can retrieve information based on semantic similarity rather than relying only on exact keyword matches.

This project explores how vector storage, nearest-neighbor algorithms, and semantic retrieval can be implemented using Java.

## 🎯 Learning Outcomes

- Understanding vector embeddings and similarity search.
- Implementing nearest-neighbor search algorithms.
- Comparing exact and approximate nearest-neighbor approaches.
- Working with Java HTTP servers and web interfaces.
- Understanding semantic retrieval and the RAG architecture.
- Integrating external embedding or language-model services when configured.

## 🔮 Future Improvements

- Persistent storage for vectors and documents.
- REST API improvements and request validation.
- Benchmarking search algorithms on larger datasets.
- Improved error handling and logging.
- Docker support and automated testing.

## 👨‍💻 Author

**YOUR NAME**

GitHub: https://github.com/YOUR_USERNAME

---

*This project was developed for learning and experimentation with vector databases, similarity search, and semantic retrieval.*
