from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords as STP
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

class  naturalCleanTxt (BaseEstimator,TransformerMixin):
    """Class for Normalise Txt cols from DataFrame, with nltk lemmatizer & stopwords"""

    def __init__(self,language="english"):
        """ Clean text from DataFrame 
        
        Parameters
        ----------
        values : 

        language for stop words : english,french ...
        
        Returns
        -------
        new dataFrame with cols cleanned """

        self.language = language


    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        lemmatizer = WordNetLemmatizer()
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
 
        stopwords = set(STP.words(f'{self.language}'))
    
        ########### lower cols used for analyse ###########

        for col in X :
            X[col] = X[col].apply(lambda x : str(x).lower() )

           
        ########### clean stop words col title ###########
            
            count = 0
            for item in X[col]:

                filtered_sentence2 = []
                word_list = nltk.word_tokenize(item)
                
                for v in word_list:
                    w = lemmatizer.lemmatize(v)
                    if w not in stopwords:
                        if len(w) > 2 :
                            filtered_sentence2.append(w)

                selected_sentence2 = ' '.join([w for w in filtered_sentence2])
                X[col][count] = selected_sentence2
                count +=1   

        return X


class  naturalSparc (BaseEstimator,TransformerMixin):
    """Class for implement the tf-idf function in a pipeline"""

    def __init__(self,language="english"):
        """ vectorize text for Ml models 
        
        Parameters
        ----------
        values : 

        language for stop words : english,french ...
        
        Returns
        -------
        ends of Preprocessing Pipeline, ready for modelization """
        
        self.language = language
        self.tfidf = TfidfVectorizer(stop_words=self.language)

    def fit(self,X,y=None):
        self.tfidf.fit(X.iloc[:, 0])
        return self

    def transform(self,X,y=None):
        return self.tfidf.transform(X.iloc[:, 0])

