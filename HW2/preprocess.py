from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import string
from nltk.stem import SnowballStemmer

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    # Begin your code (Part 0)
    '''
    import string
    from nltk.stem import SnowballStemmer
    text='I loved beach 333<br />,Dinic algorithm is thE best.'
    '''
    preprocessed_text=text.replace("<br />"," ")
    #preprocessed_text = remove_stopwords(preprocessed_text)
    preprocessed_text=preprocessed_text.lower()
    preprocessed_text="".join([i for i in preprocessed_text if i not in string.punctuation])
    e_stemmer=SnowballStemmer(language='english')
    preprocessed_text=preprocessed_text.split()
    preprocessed_text=[e_stemmer.stem(i) for i in preprocessed_text]
    preprocessed_text=" ".join(preprocessed_text)
    # End your code
    
    return preprocessed_text
'''
text='He respected Robert Tarjan <br /> and Tarjan algorithm is the best 77777.'
print(preprocessing_function(text))
'''