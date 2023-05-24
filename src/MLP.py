##### IMPORT PACKAGES
# system tools
import os

# data munging tools
import pandas as pd
from joblib import dump, load

# Machine learning stuff
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics

def main():
    
    ##### READ DATA
    # get filename
    filename = os.path.join(
        "in",  
        "fake_or_real_news.csv")

    # load data
    data = pd.read_csv(
        filename, 
        index_col = 0)

    # extract needed columns from data frame
    X = data["text"]
    y = data["label"]

    ##### TRAIN-TEST SPLIT

    X_train, X_test, y_train, y_test = train_test_split(
        X, # inputs for the model
        y, # classification labels
        test_size = 0.2,   # create an 80/20 train/test split
        random_state = 42) # random state for reproducibility

    ##### VECTORIZE
    vectorizer = TfidfVectorizer(
        ngram_range = (1, 2), # unigrams and bigrams (1 word and 2 word units)
        lowercase =  True, # don't distinguish between e.g. words at start vs middle of sentence
        max_df = 0.95, # remove very common words
        min_df = 0.05, # remove very rare words
        max_features = 500) # keep only top 500 features

    # first we fit the vectorizer to the training data...
    X_train_feats = vectorizer.fit_transform(X_train)

    #... then transform our test data
    X_test_feats = vectorizer.transform(X_test)

    # get feature names if needed
    feature_names = vectorizer.get_feature_names_out()

    ##### CLASSIFY & PREDICT
    # define classifier
    classifier = MLPClassifier(
        activation = "logistic", # use sigmoid neurons
        hidden_layer_sizes = (20,), # 1 hidden layer of 20 neurons
        max_iter = 1000, # max number of attempts to converge
        random_state = 42) # reproducibility

    # fit classifier
    classifier.fit(
        X_train_feats, 
        y_train)

    # get predictions
    y_pred = classifier.predict(
        X_test_feats)

    # evaluate
    classifier_metrics = metrics.classification_report(
        y_test, 
        y_pred)

    ##### SAVE MODELS & EVALUATION
    # DEFINE PATHS
    classifier_path = os.path.join(
        "models",
        "MLP_classifier.joblib")

    vectorizer_path = os.path.join(
        "models",
        "MLP_vectorizer.joblib")

    MLP_eval_path = os.path.join(
        "out",
        "MLP_eval.txt")

    # SAVE
    dump(
        classifier, 
        classifier_path)

    dump(
        vectorizer, 
        vectorizer_path)

    with open(MLP_eval_path, 'w') as f:
        f.write(classifier_metrics)

if __name__ == "__main__":
    main()