# ArXiv Text Classification with FNN(MLP)

## Project Overview
This project implements a text classification system for arXiv paper abstracts using Multi-Layer Perceptron (MLP) neural networks. It compares two different text vectorization techniques: TF-IDF and Count Vectorization.

## Prerequisites
- Python 3.x
- Required libraries:
  - pandas
  - scikit-learn
  - nltk
  - matplotlib

It is recommended to initilize and activate a virtual environment.
To install dependencies:
```bash
pip install -r requirements.txt
```

Download required NLTK resources:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

## Project Structure for FNN
- `constant.py`: Contains data paths and constants
- `preprocess_util.py`: Text preprocessing and data loading utilities
- `MLP_model.py`: Definition of the neural network model class
- `tf-idf.py`: Script to run models with TF-IDF vectorization
- `count_vec.py`: Script to run models with Count vectorization
- `plot-charts.py`: Utilities to visualize model training results

## Model Parameters
The neural network models use the following configurations:
- Vectorization types: TF-IDF and Count Vectorizer
- Max features: 5000
- Hidden layer sizes: 
  - (50,)
  - (100,)
  - (200,)
  - (50, 25)
  - (100, 50)
  - (200, 100)
  - (100, 50, 25)
- Training parameters:
  - max_iter: 100
  - early_stopping: False
  - n_iter_no_change: 10
  - tol: 0.0001
  - random_state: 42
  - solver: "adam"

The training data should contain at least the following columns:
- "abstract": The text of the paper abstract
- "label": The classification label

## Running the Models
All commands should be run from the FNN directory:

### TF-IDF Vectorization
```bash
cd FNN
python tf-idf.py
```

### Count Vectorization
```bash
cd FNN
python count_vec.py
```

## Output and Results
Results are automatically saved in a directory structure organized by vectorizer type and hidden layer configuration:

```
├── tfidf/
│   ├── (50,)/
│   │   ├── loss.csv
│   │   ├── accuracy.csv
│   ├── (100,)/
│   │   ├── loss.csv
│   │   ├── accuracy.csv
│   ├── ...
├── count/
│   ├── (50,)/
│   │   ├── loss.csv
│   │   ├── accuracy.csv
│   ├── ...
```

Each model configuration directory contains:
- `loss.csv`: Training loss values for each epoch
- `accuracy.csv`: Validation accuracy

## Visualizing Results
To visualize the training loss curves:

```bash
cd FNN
python plot-charts.py
```

You can modify the `plot-charts.py` file to uncomment either:
- `plot_loss_curves()`: To create individual plots for each configuration
- `plot_combined_loss_curves()`: To create a combined plot for all configurations

Generated plots will be saved as PNG files in their respective directories or at the root level.

## Preprocessing Options
The project includes two text preprocessing approaches:
- With lemmatization: Reduces words to their base form
- Without lemmatization: Basic cleaning without reducing word forms

This can be controlled with the `with_lem` variable in the execution scripts.

## Extending the Project
- To add new hidden layer configurations, modify the `hidden_layer_sizes` list in the execution scripts
- To change the vectorizer parameters, modify the `MLPModel` class in `MLP_model.py`
