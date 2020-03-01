import nltk
from utils import json_reader
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import time
from datetime import datetime
import pickle as pkl
from nltk.tokenize import ToktokTokenizer
from sklearn.metrics import confusion_matrix, f1_score
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

train_file = 'ass2_data/train.json'
test_file = 'ass2_data/test.json'

# # visualize the data ~ word-cloud
# star5_words = ' '.join(list(X[Y==5]))
# wc = WordCloud(width=512, height=512).generate(star5_words)
# plt.figure(figsize=(10, 8), facecolor='k')
# plt.imshow(wc)
# plt.show()
# print('ok')


def data_loader(filename):
    start = time.time()
    X, Y = [], []
    data_gen = json_reader(filename)
    for i, sample in enumerate(data_gen):
        review = sample['text']
        # review = nltk.word_tokenize(review)
        stars = sample['stars']
        X.append(review)
        Y.append(stars)
        # if len(Y) == 5000:
        #     break
    df = pd.DataFrame({'text': X, 'stars': Y})
    print('Time taken = {}'.format(time.time() - start))
    return df


def preprocess(data):
    X, Y = [], []
    toktok = ToktokTokenizer()
    for index, review in data.iterrows():
        if (index+1) % 100000 == 0:
            print(index+1)
        # words = nltk.word_tokenize(review['text'])
        tokens = toktok.tokenize(review['text'].lower())
        X.append(tokens)
        # X.append(nltk.word_tokenize(review['text']))
        Y.append(int(review['stars'] - 1))
        # if len(Y) == 10000:
        #     break
    df_new = pd.DataFrame({'text': X, 'stars': Y})
    return df_new


def preprocess_advanced(data):
    X, Y = [], []
    toktok = ToktokTokenizer()
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    for index, review in data.iterrows():
        if (index+1) % 100000 == 0:
            print(index+1)
        # words = nltk.word_tokenize(review['text'])
        tokens = toktok.tokenize(review['text'].lower())
        # tokens = word_tokenize(doc.lower())
        stopped_tokens = filter(lambda token: token not in en_stop, tokens)
        stemmed_tokens = map(lambda token: p_stemmer.stem(token), stopped_tokens)
        # if not return_tokens:
        #     return ' '.join(stemmed_tokens)
        # return list(stemmed_tokens)

        X.append(list(stemmed_tokens))
        # X.append(nltk.word_tokenize(review['text']))
        Y.append(int(review['stars'] - 1))
        if len(Y) == 1000:
            break
    df_new = pd.DataFrame({'text': X, 'stars': Y})
    return df_new


def parameter_estimation(train_df):
    num_classes = len(set(train_df.stars))
    # Create vocabulary
    print('{} Creating vocabulary ...'.format(datetime.now()))
    vocab = {}
    word_count = 0
    for index, review in train_df.iterrows():
        for word in review['text']:
            if word not in vocab:
                vocab[word] = word_count
                word_count += 1
    vocab_size = len(vocab)
    print('Size of vocabulary = {}'.format(vocab_size))

    # Parameter estimation
    print('{} Parameter estimation ...'.format(datetime.now()))
    frequencies = np.zeros((vocab_size, num_classes))
    phiy = np.zeros(num_classes)

    for index, review in train_df.iterrows():
        if (index + 1) % 100000 == 0:
            print(index + 1)
        phiy[review['stars']] += 1
        rating = review['stars']
        for word in review['text']:
            index = vocab[word]
            frequencies[index, rating] += 1

    sum_freq = sum(frequencies)
    params = np.zeros_like(frequencies)
    for i in range(vocab_size):
        params[i] = (frequencies[i] + 1) / (sum_freq + vocab_size)
        # params[word] = (params[word]) / (sum_freq)
    phiy = phiy / sum(phiy)

    return params, phiy, vocab, vocab_size


def predict_nb(data):
    pred_arr = np.zeros(data.shape[0])
    for index, review in data.iterrows():
        if (index + 1) % 10000 == 0:
            print(index + 1)
        posterior = np.zeros(num_classes)
        for star in range(num_classes):
            prior = np.log(phiy[star])
            likelihood = 0
            for word in review['text']:
                if word not in vocab:
                    p_word = 1 / vocab_size  # TODO: change needed here
                else:
                    word_index = vocab[word]
                    p_word = params[word_index, star]
                likelihood += np.log(p_word)
            posterior[star] = prior + likelihood
        pred_arr[index] = np.argmax(posterior)

    return pred_arr


if __name__ == '__main__':

    # Load data
    print('{} Loading training data ...'.format(datetime.now()))
    train_df = data_loader(train_file)

    print('{} Loading test data ...'.format(datetime.now()))
    test_df = data_loader(test_file)

    # Pre-processing
    print('{} ----------------- Pre-processing -----------------'.format(datetime.now()))
    start = time.time()
    train_df = preprocess(train_df)
    print('Training data preprocessing time = {}'.format(time.time() - start))
    start = time.time()
    test_df = preprocess(test_df)
    print('Test data preprocessing time = {}'.format(time.time() - start))

    # Parameter estimation
    print('{} --------------- Training ------------------'.format(datetime.now()))
    num_classes = len(set(train_df.stars))
    start = time.time()
    params, phiy, vocab, vocab_size = parameter_estimation(train_df)
    print('Time taken = {}'.format(time.time() - start))

    # # Evaluation on train data
    # print('{} Evaluating on training data ...'.format(datetime.now()))
    # start = time.time()
    # train_pred = predict_nb(train_df)
    # train_acc = accuracy_score(train_pred, train_df.stars)
    # print('Time taken = {}'.format(time.time() - start))
    # print('Train set accuracy = {}'.format(train_acc))

    # Evaluation on test data
    print('{} ------------------ Evaluating on test data --------------------'.format(datetime.now()))
    start = time.time()
    test_pred = predict_nb(test_df)
    test_acc = accuracy_score(test_pred, test_df.stars)
    print('Time taken = {}'.format(time.time() - start))
    print('{} Test set accuracy = {}'.format(datetime.now(), test_acc))

    # Random guessing
    print('{} ------------------- Random guessing ---------------------'.format(datetime.now()))
    start = time.time()
    pred_arr = np.zeros(test_df.shape[0])
    for index, review in test_df.iterrows():
        pred_arr[index] = np.random.randint(0, num_classes)
    test_acc = accuracy_score(pred_arr, test_df.stars)
    print('Time taken = {}'.format(time.time() - start))
    print('{} Test set accuracy = {}'.format(datetime.now(), test_acc))

    # Majority prediction
    majority = train_df['stars'].mode()[0]
    print('{}------------------- Majority prediction ---------------------'.format(datetime.now()))
    print('Majority class = {}'.format(majority))
    start = time.time()
    pred_arr = np.zeros(test_df.shape[0])
    for index, review in test_df.iterrows():
        pred_arr[index] = majority
    test_acc = accuracy_score(pred_arr, test_df.stars)
    print('Time taken = {}'.format(time.time() - start))
    print('{} Test set accuracy = {}'.format(datetime.now(), test_acc))

    # Confusion matrix
    print('Confusion matrix:')
    print(confusion_matrix(y_true=test_df.stars, y_pred=test_pred))

    # f1-scores
    f1_scores = f1_score(y_true=test_df.stars, y_pred=test_pred, average=None)
    macro_score = f1_score(y_true=test_df.stars, y_pred=test_pred, average='macro')
    print('F1-scores: ')
    for i, score in enumerate(f1_scores):
        print('Class {}: {}'.format(i+1, score))
    print('Macro F1-score = {}'.format(macro_score))





