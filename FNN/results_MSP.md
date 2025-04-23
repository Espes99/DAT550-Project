### Default model
max-iterations set to 100
```python
model = MLPClassifier(
    hidden_layer_sizes=self.hidden_layer_sizes,
    max_iter=self.max_iter,
    early_stopping=False,
    n_iter_no_change=10,
    tol=0.0001,
    random_state=42,
    verbose=True,
    solver="adam"
)
```

#### TFIDF lemmatization
`'WITH LEM0 .7859375', 'WITHOUT LEM0 .7805'`
*With lemmatization => 78.59%* 
*Without lemmatization => 78.05%*

#### Count lemmatization
`'WITH LEM0 .7940625', 'WITHOUT LEM0 .7778125'`
*With lemmatization => 79.41%* 
*Without lemmatization => 77.78%*

### Hidden Layers
```python
hidden_layer_sizes = [(50,), (100,), (200,), (50, 25), (100, 50), (200, 100), (100, 50, 25)]
```
#### TFIDF

| Hidden Layer Structure | Accuracy |
| ---------------------- |----------|
| (50,)                  | 0.775703 |
| (100,)                 | 0.777734 |
| (200,)                 | 0.780781 |
| (50, 25)               | 0.765547 |
| (100, 50)              | 0.768281 |
| (200, 100)             | 0.780938 |
| (100, 50, 25)          | 0.77219  |
#### Count
| Hidden Layer Structure | Accuracy |
| --- |----------|
| (50,) | 0.783438 |
| (100,) | 0.79219  |
| (200,) | 0.798359 |
| (50, 25) | 0.778516 |
| (100, 50) | 0.797969 |
| (200, 100) | 0.801719 |
| (100, 50, 25) | 0.79586  |
