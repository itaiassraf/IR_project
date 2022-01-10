from flask import Flask, request, jsonify
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
#from google.cloud import storage
import csv
from contextlib import closing


import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

nltk.download('stopwords')

from inverted_index_colab import *
from inverted_index_gcp import *
import math


#To each doc id has a title. So when we will want to return results in our search model, we will return (doc_id,title)
id_title=pd.read_pickle('/home/itaias/postings/docid_title.pickle')

#We create 3 indexes: Body,Anchor and Title. We put it in our drive and reading it from the drive.
title_index = InvertedIndex.read_index('/home/itaias/postings/postings_title_without_filter/_title postings_gcp', 'index')
body_index=InvertedIndex.read_index('/home/itaias/postings/postings_body/postings_gcp_body', 'index')
anchor_index=InvertedIndex.read_index('/home/itaias/postings/postings_anchor_without_filter/postings_gcp_anchor', 'index')

#path of the location of the indexes in our drive
path_body='/home/itaias/postings/postings_body/postings_gcp_body/'
path_title='/home/itaias/postings/postings_title_without_filter/_title postings_gcp/'
path_anchor='/home/itaias/postings/postings_anchor_without_filter/postings_gcp_anchor/'

#for each doc_id see how much views he has
page_viewsdict=pd.read_pickle('/home/itaias/postings/pageviews-202108-user.pkl')

#for each doc id see how much page_rank score he has
with open('/home/itaias/postings/page_rank.csv','r') as f:
  reader1=csv.reader(f)
  data=list(reader1)
  dict_pagerank=dict([(int(i),float(j)) for i,j in data])
  




class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)



app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer
def read_posting_list(inverted, w,file_name):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    locs = [(file_name + lo[0], lo[1]) for lo in locs]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list


RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))
def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.
    
    Parameters:
    -----------
    text: string , represting the text to tokenize.    
    
    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens =  [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in stopwords_frozen]    
    return list_of_tokens 

                        

def change_dict(list_tuples_id_score):
  doc_id_title={}
  for doc_id,score in list_tuples_id_score:
    doc_id_title[doc_id]=id_title[doc_id]
  
  return list(doc_id_title.items())

def get_candidate_documents_and_scores(query_to_search,index,path):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.
    
    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.'). 
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score. 
    """
    
    DL=index.DL
    words=list(index.term_total.keys())
    tokens=tokenize(query_to_search)
    query_counter=Counter(tokens)
    query_counterd=dict(query_counter)
    candidates = {}
    N = len(DL)        
    for term in np.unique(tokens):

        if term in words: 

            list_of_doc = read_posting_list(index,term,path)
            normlized_tfidf=[]
            for doc_id, freq in list_of_doc:
              if (doc_id,freq)==(0,0):
                 continue

              formula=(freq/DL[doc_id]) * math.log(N/index.df[term],10) * query_counterd[term]   
              id_tfidf=(doc_id,formula)            
              normlized_tfidf.append(id_tfidf)          
                        
            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id,term)] = candidates.get((doc_id,term), 0) + tfidf               
        
    return candidates


def cosine_similarity(search_query, index,path):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores 
    key: doc_id
    value: cosine similarity score
    
    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores
    
    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    
    dict_cosine_sim={}
    candidates = get_candidate_documents_and_scores(search_query, index,path)
    for doc_id_term, normalized_tfidf in candidates.items():
          dict_cosine_sim[doc_id_term[0]] = normalized_tfidf / (len(search_query) * index.DL[doc_id_term[0]])
    
    return dict_cosine_sim



def get_top_n(sim_dict,N=100):
    """ 
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores 
   
    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3
    
    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    
    return sorted([(doc_id,np.round(score,5)) for doc_id, score in sim_dict.items()], key = lambda x: x[1],reverse=True)[:N]

def Binary_Docs(query_to_search, index, file_name):
    
    doc_id_freq = {}
    words=list(index.term_total.keys())
    tokens=tokenize(query_to_search)

    for term in np.unique(tokens):
        if term in words: 

          list_of_doc = read_posting_list(index, term,file_name) 
          for doc_id, freq  in list_of_doc:
              if doc_id in doc_id_freq:
                  doc_id_freq[doc_id] += 1
              else:
                  doc_id_freq[doc_id] = 1
                        
    return sorted(doc_id_freq.items(), key=lambda x: x[1], reverse=True)

#Page rank method-return to each wiki id his grade
def page_rank(wiki_id,page_rank_dict):
  lst_pagerank=[]
  for wikiid in wiki_id:
    try:
      lst_pagerank.append((wikiid,page_rank_dict[wikiid]))
    except:
      lst_pagerank.append((wikiid,0))
    
  
  return lst_pagerank


#return only the page rank score/amount of his views for wiki id (using it to the search method by page rank/by page_views)
def return_wiki_from_pagerank_or_page_view(res):
    lst_wikis_id=[]
    for doc_id,pr_score in res:
        lst_wikis_id.append(pr_score)
        
    return pr_score

#reutrn to each wiki id his grade
def page_views(wiki_id,page_views_dict):
  lst_pageviews=[]
  for wikiid in wiki_id:
    try:
      lst_pageviews.append((wikiid,page_views_dict[wikiid]))
    except:
      lst_pageviews.append((wikiid,0))
      
  return lst_pageviews


#For Search method
def BM25_search_body3(queries,N=100):
    
    #aveage doc length
    avg_doc_length_of_all_corpus = sum(body_index.DL.values()) / len(body_index.DL)
    
    #size of the corpus
    size_corpus = len(body_index.DL)
    
    tokens=tokenize(queries)
    all_docs_distinct = []
    term_docid_freq = {}
    
    #for on all the terms in the query
    for term in tokens:
            
            if term in body_index.term_total:
                
              list_docid_tf_foreach_term = read_posting_list(body_index, term,'/home/itaias/postings/postings_body/postings_gcp_body/')
              lst_docid=[]
              
              #getting a list of doc id to each term 
              #getting a dictionary that to each (doc_id,term) his tf-term frequency
              for doc_id,freq in list_docid_tf_foreach_term:
                  term_docid_freq[(term, doc_id)] = freq
                  lst_docid.append(doc_id)
              
              all_docs_distinct+=lst_docid
    
    #getting only distinct docs
    all_docs_distinct= set(all_docs_distinct)

    def BM25_score_docid_query(query, doc_id,k1=1.5, b=0.75):
        
        #In this function for each term in the query and to each doc that relevantive to the query (that term in this document)
        #We calculate the bm25 score between the document to all the terms in the query!
        
        idf = calc_idf(query)
        bm25 = 0
        for term in query:
            
            #by the formula like in homework 4 !
            if (term,doc_id) in term_docid_freq:
                freq=term_docid_freq[(term,doc_id)]
                first=(k1+1)* freq
                secondpart=query[term] * idf[term]
                thirdpart=freq + k1 * (1 - b + b * (body_index.DL[doc_id] / avg_doc_length_of_all_corpus))
                                                      
                bm25 += secondpart * (first  / thirdpart)
        return bm25

    def calc_idf(query):
        
        #calculate bm25 by the formula we learn at the class
        idf = {}
        for term in query:
            if term not in body_index.term_total.keys():  
                idf[term] = 0
            else:

                mone= size_corpus - body_index.df[term] + 0.5
                mechane=body_index.df[term] + 0.5 + 1
                idf[term] = math.log((mone / mechane)+1)
            
        return idf
    
    doc_id_bm25=[]
    for doc_id in all_docs_distinct:
        doc_id_bm25.append((doc_id,BM25_score_docid_query(dict(Counter(tokens)),doc_id,1.5,0.75)))
    
    doc_id_bm25=sorted(doc_id_bm25, key=lambda x: x[1], reverse=True)[:100]

    res=change_dict(doc_id_bm25)
    
    return res

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    if len(tokenize(query))==1:
        res=Binary_Docs(query, title_index, path_title)
        res=change_dict(res)
    else: 
        res=BM25_search_body3(query, N=100)
    # END SOLUTION
    return jsonify(res)
   

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res=cosine_similarity(query,body_index,path_body)
    res=get_top_n(res,N=100)
    res=change_dict(res)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res=Binary_Docs(query,title_index,path_title)
    res=change_dict(res)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res=Binary_Docs(query,anchor_index,path_anchor)
    res=change_dict(res)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res=page_rank(wiki_ids,dict_pagerank)
    res=return_wiki_from_pagerank_or_page_view(res)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])


def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res=page_views(wiki_ids,page_viewsdict)
    res=return_wiki_from_pagerank_or_page_view(res)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
