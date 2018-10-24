import pandas
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

demo_loading_data = False
demo_data_frame = False
demo_data_split = False
demo_count_vectors = True

#####################################  Loading Data from Corpus File #####################################

# load the dataset
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
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

if demo_data_frame:
    print('data_frame: ')
    print(trainDF)

################################  Split the Dataset into Training & Validation ################################

# split the dataset into training and validation datasets
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
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
array_of_texts = [" ".join(sentence_array) for sentence_array in trainDF.text.values]
if demo_count_vectors:
    print('count_vect: ')
    print(count_vect)
count_vect.fit(array_of_texts)
if demo_count_vectors:
    print('count_vect after fit: ')
    print(count_vect)

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform([" ".join(sentence_array) for sentence_array in train_x])
xvalid_count =  count_vect.transform([" ".join(sentence_array) for sentence_array in valid_x])

if demo_count_vectors:
    print('#### xtrain_count: ')
    print(xtrain_count)
    print('#### xvalid_count: ')
    print(xvalid_count)