from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class MLPModel:
    def __init__(self, vectorizer_type, hidden_layer_sizes=(100,), max_features=5000, max_iter=100):
        self.vectorizer_type = vectorizer_type
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_features = max_features
        self.max_iter = max_iter

        # Set the vectorizers defined in the desciription
        if vectorizer_type.lower() == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        elif vectorizer_type.lower() == 'count':
            self.vectorizer = CountVectorizer(max_features=self.max_features)

        # Initialize MLP classifier model
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            early_stopping=False,
            n_iter_no_change=10,
            tol=0.0001,
            random_state=42,
            verbose=True
        )

    def fit(self, X_train, y_train):
        print("Fitting the model: ", self.vectorizer_type)
        print("With a hidden layer size of: ", self.hidden_layer_sizes)
        # Transform the data using the vectorizer
        X_train_vec = self.vectorizer.fit_transform(X_train)

        # Train the model
        self.model.fit(X_train_vec, y_train)

        return self

    def evaluate(self, X_val, y_val):
        X_val_vec = self.vectorizer.transform(X_val)

        # Make predictions
        y_val_pred = self.model.predict(X_val_vec)

        # Calculate metrics
        accuracy_val = accuracy_score(y_val, y_val_pred)
        # If we are to run long continous tests, we should write results to a file

        # We can possibly append a confusion matrix and classification report to the dict here if it is needed
        # report = classification_report(y_val, y_val_pred, output_dict=True)

        print(f"Validation Accuracy: {accuracy_val:.4f}")
        # print(f"\n{self.vectorizer_type.upper()} Model Validation Classification Report:")
        # print(report)

        return {
            'accuracy': accuracy_val,
            # 'report': report
        }

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)