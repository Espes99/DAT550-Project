# Bag of Words Document Classification using Feedforward Neural Network and Recurrent Neural Network

This project compares the performance of **Feedforward Neural Networks (FNNs)** and **Recurrent Neural Networks (RNNs)** for classifying scientific abstracts from the ArXiv-10 dataset into predefined topics (e.g., 'cs', 'math', 'physics').  
The goal is to evaluate how different architectures (FNN vs. RNN) and text representations (Bag-of-Words vs. word embeddings) impact classification accuracy.

## Key Features
- **FNN Experiments**:  
  - Uses Bag-of-Words (via CountVectorizer/TF-IDF) with a Multi-Layer Perceptron (MLP).
- **RNN Experiments**:  
  - Leverages pretrained embeddings (GloVe/FastText) with LSTM/GRU architectures and attention mechanisms.
- **Results**:  
  - Achieves up to **80.2% accuracy** with FNNs and **83% accuracy** with RNNs.  

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Espes99/DAT550-Project.git
   cd DAT550-Project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Detailed Instructions

- **FNN Setup**: See [FNN/README.md](FNN/README.md) for running Bag-of-Words experiments.
- **RNN Setup**: See [RNNs/README.md](RNNs/README.md) for running RNN/LSTM/GRU experiments.

## Dataset

The ArXiv-10 dataset consists of abstracts labeled with 10 scientific categories.  
Preprocessing steps include:
- Tokenization
- Stopword removal
- Lemmatization


## Important notes:
- This project uses torch version 2.3.0 because that is the latest version compatible with torchtext version 0.18.0
