# Count Vectorizer and Naive Bayes Classifier for Text Classification

This project demonstrates the use of Count Vectorizer and Naive Bayes Classifier for text classification tasks. The notebook includes data preprocessing steps such as tokenization, stemming, and lemmatization using NLTK.

## Features

- **Count Vectorizer**: Converts a collection of text documents to a matrix of token counts.
- **Naive Bayes Classifier**: A probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
- **Data Preprocessing**: Includes tokenization, stemming, and lemmatization using NLTK.

## Installation

To run this project, you will need to install the following libraries:

```bash
pip install numpy pandas scikit-learn nltk
```

## Usage

1. **Import the necessary libraries:**

    ```python
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    import nltk
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.corpus import wordnet
    ```

2. **Download NLTK data:**

    ```python
    nltk.download("wordnet")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    ```

3. **Load your dataset and preprocess the text:**

    ```python
    # Example of loading a dataset
    df = pd.read_csv('your_dataset.csv')

    # Example of text preprocessing
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    def preprocess(text):
        tokens = word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
        return ' '.join(stemmed_tokens)

    df['processed_text'] = df['text_column'].apply(preprocess)
    ```

4. **Vectorize the text and train the classifier:**

    ```python
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label_column']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Evaluate the model
    accuracy = classifier.score(X_test, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    ```

## Contributing

If you wish to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature-branch`)
6. Open a pull request
