# part1
import pickle
import numpy as np
# part2
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
# part3
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM

def chatRobot():
    # part1
    # retrieve training data
    with open('./Deep-Learning-for-NLP-Creating-a-Chatbot-master/train_qa.txt', 'rb') as f:
        train_data = pickle.load(f)

    # retrieve test data
    with open('./Deep-Learning-for-NLP-Creating-a-Chatbot-master/test_qa.txt', 'rb') as f:
        test_data = pickle.load(f)

    # Number of training instances
    len(train_data)

    # Example of one of the instances
    #train_data[10]
    ' '.join(train_data[10][0])
    ' '.join(train_data[10][1])
    #train_data[10][2]

    # First we need to create a vocabulary with our data
    # For this we will use the training data only to - On the video it uses both
    # train and test
    # Might have to use training and test later, as the dataset has very
    # few words
    vocab = set()
    for story, question, answer in train_data:
        vocab = vocab.union(set(story))  # Set returns unique words in the sentence
        # Union returns the unique common elements from a two sets
        vocab = vocab.union(set(question))

    vocab.add('no')
    vocab.add('yes')

    #print(vocab)

    # Calculate len and add 1 for Keras placeholder - Placeholders are used to feed in the data to the network.
    # They need a data type, and have optional shape arguements.
    # They will be empty at first, and then the data will get fed into the placeholder
    vocab_len = len(vocab) + 1

    # Now we are going to calculate the longest story and the longest question
    # We need this for the Keras pad sequences.
    # Keras training layers expect all of the input to have the same length, so
    # we need to pad
    all_data = test_data + train_data
    all_story_lens = [len(data[0]) for data in all_data]
    max_story_len = (max(all_story_lens))
    max_question_len = max([len(data[1]) for data in all_data])

    #  Part2 Vectorizing the data

    # Create an instance of the tokenizer object:
    tokenizer = Tokenizer(filters=[])
    tokenizer.fit_on_texts(vocab)

    # Dictionary that maps every word in our vocab to an index
    # It has been automatically lowercased
    # This tokenizer can give different indexes for different words depending on when we run it
    print(tokenizer.word_index)

    # Tokenize the stories, questions and answers:
    train_story_text = []
    train_question_text = []
    train_answers = []

    # Separating each of the elements
    for story, question, answer in train_data:
        train_story_text.append(story)
        train_question_text.append(question)
        train_answers.append(answer)

    # Coverting the text into the indexes
    train_story_seq = tokenizer.texts_to_sequences(train_story_text)

    # Create a function for vectorizing the stories, questions and answers:
    def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len,
                          max_question_len=max_question_len):
        # vectorized stories:
        X = []
        # vectorized questions:
        Xq = []
        # vectorized answers:
        Y = []

        for story, question, answer in data:
            # Getting indexes for each word in the story
            x = [word_index[word.lower()] for word in story]
            # Getting indexes for each word in the story
            xq = [word_index[word.lower()] for word in question]
            # For the answers
            y = np.zeros(len(word_index) + 1)  # Index 0 Reserved when padding the sequences
            y[word_index[answer]] = 1

            X.append(x)
            Xq.append(xq)
            Y.append(y)

        # Now we have to pad these sequences:
        return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y))

    inputs_train, questions_train, answers_train = vectorize_stories(train_data)
    inputs_test, questions_test, answers_test = vectorize_stories(test_data)
    print(inputs_train[0])
    print(train_story_text[0])
    print(train_story_seq[0])

    # part3 building the network

    # We need to create the placeholders
    # The Input function is used to create a keras tensor
    # PLACEHOLDER shape = (max_story_len,batch_size)
    # These are our placeholder for the inputs, ready to recieve batches of the stories and the questions
    input_sequence = Input((max_story_len,))  # As we dont know batch size yet
    question = Input((max_question_len,))

    # Create input encoder M:
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_len, output_dim=64))  # From paper
    print(input_encoder_m.add(Dropout(0.3)))

    # Outputs: (Samples, story_maxlen,embedding_dim) -- Gives a list of the lenght of the samples where each item has the
    # lenght of the max story lenght and every word is embedded in the embbeding dimension

    # Create input encoder C:
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_len, output_dim=max_question_len))  # From paper
    print(input_encoder_c.add(Dropout(0.3)))

    # Outputs: (samples, story_maxlen, max_question_len)

    # Create question encoder:
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_len, output_dim=64, input_length=max_question_len))  # From paper
    print(question_encoder.add(Dropout(0.3)))

    # Outputs: (samples, question_maxlen, embedding_dim)

    # Now lets encode the sequences, passing the placeholders into our encoders:
    input_encoded_m = input_encoder_m(input_sequence)
    input_encoded_c = input_encoder_c(input_sequence)
    question_encoded = question_encoder(question)

    # Use dot product to compute similarity between input encoded m and question
    # Like in the paper:
    match = dot([input_encoded_m, question_encoded], axes=(2, 2))
    match = Activation('softmax')(match)

    # For the response we want to add this match with the ouput of input_encoded_c
    response = add([match, input_encoded_c])
    response = Permute((2, 1))(response)  # Permute Layer: permutes dimensions of input

    # Once we have the response we can concatenate it with the question encoded:
    answer = concatenate([response, question_encoded])

    print(answer)

    # Reduce the answer tensor with a RNN (LSTM)
    answer = LSTM(32)(answer)

    # Regularization with dropout:
    answer = Dropout(0.5)(answer)
    # Output layer:
    answer = Dense(vocab_len)(answer)  # Output shape: (Samples, Vocab_size) #Yes or no and all 0s

    # Now we need to output a probability distribution for the vocab, using softmax:
    answer = Activation('softmax')(answer)

    # Now we build the final model:
    model = Model([input_sequence, question], answer)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # Categorical instead of binary cross entropy as because of the way we are training
    # we could actually see any of the words from the vocab as output
    # however, we should only see yes or no

    print(model.summary())

    history = model.fit([inputs_train, questions_train], answers_train, batch_size=32, epochs=2,
                        validation_data=([inputs_test, questions_test], answers_test))

    print('-------------------------')
    print(history.history)
    print('-------------------------')

    filename = 'Z_chatbot_100_epochs.h5'
    model.save(filename)

    # Lets plot the increase of accuracy as we increase the number of training epochs
    # We can see that without any training the acc is about 50%, random guessing
    import matplotlib.pyplot as plt
    # %matplotlib inline
    print(history.history.keys())
    # summarize history for accuracy
    plt.figure(figsize=(12, 12))

    #plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # To load a model that we have already trained and saved:
    model.load_weights('Z_chatbot_100_epochs.h5')

    # Lets check out the predictions on the test set:
    # These are just probabilities for every single word on the vocab
    pred_results = model.predict(([inputs_test, questions_test]))

    # First test data point
    test_data[0]

    # These are the probabilities for the vocab words using the 1st sentence
    pred_results[0]

    val_max = np.argmax(pred_results[0])

    for key, val in tokenizer.word_index.items():
        if val == val_max:
            k = key
    print(k)

    # See probability:
    pred_results[0][val_max]

    # Now, we can make our own questions using the vocabulary we have
    print(vocab)

    my_story = 'Sandra picked up the milk . Mary travelled left . '
    my_story.split()
    my_question = 'Sandra got the milk ?'
    my_question.split()

    # Put the data in the same format as before
    my_data = [(my_story.split(), my_question.split(), 'yes')]

    # Vectorize this data
    my_story, my_ques, my_ans = vectorize_stories(my_data)

    # Make the prediction
    pred_results = model.predict(([my_story, my_ques]))

    val_max = np.argmax(pred_results[0])

    # Correct prediction!
    for key, val in tokenizer.word_index.items():
        if val == val_max:
            k = key
    print(k)

    # Confidence
    print(pred_results[0][val_max])
def runRobot():
    print('runRobot')


if __name__ == '__main__':
    chatRobot()
    #runRobot()
