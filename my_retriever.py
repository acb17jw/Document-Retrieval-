from collections import Counter
from operator import itemgetter
import math

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting
        self.documentFrequency = {}
        self.documentCollection = set()
        
        for term in self.index:
        
            #Each term in the document collection is mapped to the number of documents containing this term
            self.documentFrequency.update( {term : len(self.index[term])})
            for i in self.index[term]:
                #Each document is added to the set of document collection
                self.documentCollection.add(i)
                
    # Method performing retrieval for specified query
    def forQuery(self, query):
        if self.termWeighting == 'tf':
            return self.smilarityScore(self.documentFrequencyWeighting(query))
        elif self.termWeighting == 'tfidf':
            return self.smilarityScore(self.tfidfWeighting(query))
        else:
            return self.smilarityScore(self.binaryWeighting(query))
            
    """
    Method performing binary term weighting.
    Returns a tuple that consists of two dictionaries
    first with weight assigned to each term in query, 
    second with weight assigned to each term in each document.
    """
    def binaryWeighting(self, query):
        weightTerms = {}
        weightQuery = {}
        
        for term in query:
        
            #Checks wether the term in the query is in the document collection
            if term in self.index:
            
                #Mapps each term within document that exists in the query with weight equal 1
                for i in self.index[term]:
                    if i in weightTerms:
                        weightTerms[i].update( {term : 1} )
                    else:
                        weightTerms[i] = {}
                        weightTerms[i][term] = 1
                        
        #Mapps each term in the query with the weight equal 1.
        for term in query:
            weightQuery.update( {term : 1} )
        return weightQuery, weightTerms
        
    """
    Method performing document frequency term weighting.
    Returns a tuple that consists of two dictionaries
    first with weight assigned to each term in query, 
    second with weight assigned to each term in each document.
    """       
    def documentFrequencyWeighting(self, query):
        weightTerms = {}
        
        for term in self.index:
    
                #Mapps each term within document with weight
                #The weight assigned to each term is the number of occurences of this term within the document.
                for i in self.index[term]: 
                    if i in weightTerms:
                        weightTerms[i].update( {term : self.index[term].get(i)} )
                    else:
                        weightTerms[i] = {}
                        weightTerms[i][term] = self.index[term].get(i)

        return query, weightTerms
        
    """
    Method performing tf.idf term weighting.
    Returns a tuple that consists of two dictionaries
    first with weight assigned to each term in query, 
    second with weight assigned to each term in each document.
    """    
    def tfidfWeighting(self, query):
        weightTerms = {}
        weightQuery = {}
                
        #Mapps each term within document with weight
        #The weight assigned to each term is the multiplication of tf and idf.
        for term in self.index:
                for i in self.index[term]:
                
                    #For each term in each document the inverse document frequency (idf) is computed. 
                    idf = math.log(len(self.documentCollection)/self.documentFrequency.get(term),10)
                    
                    #Term frequency (tf) is taken from index as the number of term occurences
                    if i in weightTerms:
                        tf = self.index[term].get(i)
                        
                        #Each term is mapped to the value of idf multiplied by tf
                        weightTerms[i].update( {term : idf*tf} )
                    else:
                        weightTerms[i] = {}
                        tf = self.index[term].get(i)
                        
                        #Each term is mapped to the value of idf multiplied by tf
                        weightTerms[i][term] = idf*tf
                        
        #Mapps each term in query that exists in the document collection with weight tf.idf
        for term in query:
            if term in self.index:
                idf = math.log(len(self.documentCollection)/self.documentFrequency.get(term),10)
                tf = query.get(term)
                weightQuery.update( {term : idf*tf} ) 
     
        return weightQuery, weightTerms
        
    """ 
    Method computing similarity between document and query as the cosine of the angle between their vectors
    It takes tuple of 2 vectors first the query vector and second the document vector
    Returns list of the documents sorted by their similarity to the query
    """
    def smilarityScore(self, spaceVectors):
    
        #Assignment of the query vector 
        query = spaceVectors[0]
        
        #Assignment of the document vector
        docs = spaceVectors[1]

        weightSum = 0
        termSum = 0
        scores = []
        
        #For each document the score (cosine of the angle between its vector and query vector) is computed
        for doc in docs:
            for elem in docs[doc]:
                if elem in query:
                    weightSum += (docs[doc].get(elem) * query.get(elem))
                termSum += (docs[doc].get(elem)**2)

            score = weightSum / math.sqrt(termSum)

            #Each score is added to the list
            scores.append( (doc, score) ) 
            
            weightSum = 0
            termSum = 0
            
        #The list is sorted by the values of scores    
        sortedScores = sorted(scores, key=itemgetter(1))
        sortedScores.reverse()

        #Takes only document ids from the score list.
        res_list = [x[0] for x in sortedScores]
        
        return res_list