### Default model
max-iterations set to 100 (usually stops before this)
```python
self.model = MLPClassifier(  
    hidden_layer_sizes=self.hidden_layer_sizes,  
    max_iter=self.max_iter,  
    early_stopping=False,  
    n_iter_no_change=10,  
    tol=0.0001,  
    random_state=42,  
    verbose=True  
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

| Hidden Layer Structure | Accuracy  |
| ---------------------- | --------- |
| (50,)                  | 0.78075   |
| (100,)                 | 0.7859375 |
| (200,)                 | 0.7936875 |
| (50, 25)               | 0.7751875 |
| (100, 50)              | 0.782625  |
| (200, 100)             | 0.793125  |
| (100, 50, 25)          | 0.7825625 |
#### Count
| Hidden Layer Structure | Accuracy |
| --- | --- |
| (50,) | 0.7858125 |
| (100,) | 0.7940625 |
| (200,) | 0.8 |
| (50, 25) | 0.7878125 |
| (100, 50) | 0.801625 |
| (200, 100) | 0.8086875 |
| (100, 50, 25) | 0.7978125 |
