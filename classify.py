import pandas
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras import layers, models, optimizers

demo_loading_data = False
demo_data_frame = False
demo_data_split = False
demo_count_vectors = False
demo_naive_bayes = True
demo_linear_classifier = True
demo_shallow_neural_net = True

#####################################  Loading Data from Corpus File #####################################

# load the dataset
print('Loading data from corpus...')
data = open('data/corpus').read()   # opens and reads from data file
labels, texts = [], []
for i, line in enumerate(data.split("\n")): # for each line
    content = line.split()  # split line into array of words
    labels.append(content[0])   # store in label/text arrays
    texts.append(content[1:])

    if demo_loading_data:
        print(content)
    if demo_loading_data and i == 2:
        break

if demo_loading_data:
    print('labels: ')
    print(labels)
    print('texts: ')
    print(texts)

#####################################  Create a Pandas Data Frame #####################################

# create a dataframe using texts and labels
print('Storing data into Pandas DataFrame...')
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

if demo_data_frame:
    print('data_frame: ')
    print(trainDF)

################################  Split the Dataset into Training & Validation ################################

# split the dataset into training and validation datasets
print('Splitting data into training and validation sets...')
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

if demo_data_split:
    print("######  training texts: ")
    print(train_x)
    print("######  training labels: ")
    print(train_y)
    print('#### length of trainig texts: ')
    print(len(train_x))
    print('#### length of trainig labels: ')
    print(len(train_y))
    print('#### length of validation texts: ')
    print(len(valid_x))
    print('#### length of validation labels: ')
    print(len(valid_y))


#####################################  Creating Count Feature Vector #####################################

# create a count vectorizer object
print('Creating count feature vectors...')
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
array_of_texts = [" ".join(sentence_array) for sentence_array in trainDF.text.values]
array_of_training_texts = [" ".join(sentence_array) for sentence_array in train_x]
array_of_validation_texts = [" ".join(sentence_array) for sentence_array in valid_x]
if demo_count_vectors:
    print('count_vect: ')
    print(count_vect)
count_vect.fit(array_of_texts)
if demo_count_vectors:
    print('count_vect after fit: ')
    print(count_vect)

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(array_of_training_texts)
xvalid_count =  count_vect.transform(array_of_validation_texts)

if demo_count_vectors:
    print('#### xtrain_count: ')
    print(xtrain_count)
    print('#### xvalid_count: ')
    print(xvalid_count)


#####################################  Creating TF-IDF Feature Vector #####################################

# word level tf-idf
print('Building word-level tf-idf feature vectors')
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(array_of_texts)
xtrain_tfidf =  tfidf_vect.transform(array_of_training_texts)
xvalid_tfidf =  tfidf_vect.transform(array_of_validation_texts)

# ngram level tf-idf
print('Building ngram-level tf-idf feature vectors')
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(array_of_texts)
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(array_of_training_texts)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(array_of_validation_texts)

# characters level tf-idf
print('Building char-level tf-idf feature vectors')
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(array_of_texts)
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(array_of_training_texts)
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(array_of_validation_texts)


#####################################  Generic Model Training Utility Function #####################################

def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    # fit the training dataset on the classifier
    if is_neural_net:
        classifier.fit(feature_vector_train, label, epochs=10)
    else:
        classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)


#####################################  Naive Bayes Model  #####################################

if demo_naive_bayes:
    # Naive Bayes on Count Vectors
    print('\nTraining Naive Bayes model on count vectors...')
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count, valid_y)
    print "NB, Count Vectors: ", accuracy

    # Naive Bayes on Word Level TF IDF Vectors
    print('\nTraining Naive Bayes model on word-level tf-idf vectors...')
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
    print "NB, WordLevel TF-IDF: ", accuracy

    # Naive Bayes on Ngram Level TF IDF Vectors
    print('\nTraining Naive Bayes model on ngram-level tf-idf vectors...')
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
    print "NB, N-Gram Vectors: ", accuracy

    # Naive Bayes on Character Level TF IDF Vectors
    print('\nTraining Naive Bayes model on char-level tf-idf vectors...')
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
    print "NB, CharLevel Vectors: ", accuracy


#####################################  Naive Bayes Model  #####################################

if demo_linear_classifier:
    # Linear Classifier on Count Vectors
    print('\nTraining Linear Classifier on count vectors...')
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count, valid_y)
    print "LR, Count Vectors: ", accuracy

    # Linear Classifier on Word Level TF IDF Vectors
    print('\nTraining Linear Classifier on word-level tf-idf vectors...')
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf, valid_y)
    print "LR, WordLevel TF-IDF: ", accuracy

    # Linear Classifier on Ngram Level TF IDF Vectors
    print('\nTraining Linear Classifier on ngram-level tf-idf vectors...')
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y)
    print "LR, N-Gram Vectors: ", accuracy

    # Linear Classifier on Character Level TF IDF Vectors
    print('\nTraining Linear Classifier on char-level tf-idf vectors...')
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars, valid_y)
    print "LR, CharLevel Vectors: ", accuracy


#####################################  Shallow Neural Net  #####################################

def create_model_architecture(input_size):
    # create input layer
    input_layer = layers.Input((input_size, ), sparse=True)

    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)

    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier

if demo_shallow_neural_net:
    print('\nTraining shallow neural net...')
    classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
    accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, valid_y, is_neural_net=True)
    print "NN, Ngram Level TF IDF Vectors",  accuracy