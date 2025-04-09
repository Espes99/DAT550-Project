import nltk
import constant as const
from MLP_model import MLPModel
from preprocess_util import read_data, preprocess_text_series

nltk.download('stopwords')
nltk.download('punkt_tab')
with_lem = True
accuracy_results = []
hidden_layer_sizes = [(50,), (100,), (200,), (50, 25), (100, 50), (200, 100), (100, 50, 25)]

# Load and preprocess data
X_train, X_val, y_train, y_val = read_data(const.TRAINPATH)
X_train_clean = preprocess_text_series(X_train, with_lem)
X_val_clean = preprocess_text_series(X_val, with_lem)

for hidden_layer_size in hidden_layer_sizes:
    # Create and train TF-IDF MLP model
    tfidf_model = MLPModel(vectorizer_type='tfidf', hidden_layer_sizes=hidden_layer_size)
    tfidf_model.fit(X_train_clean, y_train)

    # Evaluate the model
    results = tfidf_model.evaluate(X_val_clean, y_val)
    accuracy_results.append(f"Accuracy for: {hidden_layer_size} {results['accuracy']}")


print(accuracy_results)