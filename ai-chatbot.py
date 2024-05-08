import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open('data.txt', 'r', errors = 'ignore')
raw_doc = f.read()

# Convert it to lowercase
raw_doc = raw_doc.lower()

# Using punkt tokenizer
nltk.download('punkt')

# Using the wordnet dictionary
nltk.download('wordnet')
nltk.download('omw-1.4')

sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)


sentence_tokens[:5]
word_tokens[:5]


# Performing Text-PreProcessing
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# Define Greeting functions
greet_inputs = ('hello', 'hi', 'how are you?', 'hey')
greet_responses = ('hello', 'hi', 'hey there', 'hey')

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

def response(user_response):
    robol_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robol_response = robol_response + "I am sorry. Not able to understand"
        return robol_response
    else:
        robol_response = robol_response + sentence_tokens[idx]
        return robol_response

# Define chatflow
flag = True
print('Hello! I am a bot. start typing for more information. to end the conversation say bye')
while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thank you' or user_response == 'thanks':
            flag = False
            print('Bot: You are welcome')
        else:
            if greet(user_response) is not None:
                print('Bot: ' + greet(user_response))
            else:
                sentence_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print('Bot:', end=' ')
                print(response(user_response))
                sentence_tokens.remove(user_response)
    else:
        flag = False
        print('Bot: Goodbye')
