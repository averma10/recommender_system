
# coding: utf-8

# In[ ]:


######################################################################
### Library contains functions related to:
###  - calculating NDCG metrics
###  - store NDCG results on disk location
######################################################################

#import libraries

from collections import defaultdict
import numpy as np
import pickle
import os

###########################################
####  Might Need to Change this for IBCF ##
###########################################

def get_all_relevance(predictions,user_based = True):
    '''
    returns a list of prediction ratings related to user_id/item_id as returned by our KNNBasic model
    Arguments:
        predictions: all the test predictions returned by model.
        user_based: (binary) True will create dictionary with user_id, ratings pair and False will create dictionary with item_id, ratings pair.
                    Default value is True
        
    Return:
        A default dictionary object with (uid/iid,[list of predictions]) key-value pair.
    
    '''
    
    #map the prediction to each user using defaultdict(list)
    rel = defaultdict(list)
    if user_based:
        for uid, iid, true_r, est, _ in predictions:
            rel[uid].append((est))
    else:
        for uid, iid, true_r, est, _ in predictions:
            rel[iid].append((est))
            
    return rel


def dcg_at_k(r,k,method=0):
    """
    Calculate Discounted Cumulative Gain score.
    
    References:
        https://gist.github.com/bwhite/3726239
        http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf

    Arguments:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    
    r = np.asfarray(r)[:k]
    if r.size:
        if method ==0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2,r.size + 1)))
        elif method ==1:
            return np.sum(r / np.log2(np.arange(2,r.size + 2)))
        else:
            return ValueError('method must be 0 or 1.')
        
        return 0    
    

def ndcg_at_k(r, k, method=0):
    """
    Calculates Normalized Discounted Cumulative Gain score.
    
    References:
        https://gist.github.com/bwhite/3726239
        http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf

    Arguments:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        NDCG(Normalized discounted cumulative gain)
    """
    
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    
    if not dcg_max:
        return 0
    
    return dcg_at_k(r, k, method) / dcg_max


def get_ndcg_results(dfname,K=20):
    '''
    Compute NDCG@K results and store on disk location for later reference.
    Arguments:
        dfname: data frame name. Valid values ['movies','games','tv','music']
        K: (int) number of recommendations to consider
    
    Return:
        NDCG@K
        
    Note: The number of neighbors are hard-coded as of now with values [2,4,8,16,32,64]. I will device a better generic solution once time permits.
    
    '''
    k_list = [2,4,8,16,32,64]
    method = ['ubcf','ibcf']
    ndcg_results = pd.DataFrame(data=None,columns=[['k','ndcg','method']])
    loc = 0
    for method in method:
        for n in k_list:
            pred = load_predictions(method,dfname,n)
            ndcg =[]
            rel = get_all_relevance(pred)
            for key,v in rel.items():
                ndcg.append(ndcg_at_k(v,K))
            mean_ndcg = np.mean(ndcg)

            ndcg_results.loc[loc] = n,mean_ndcg,method
            loc+=1
    save_results('ndcg_{}_{}'.format(dfname,K),ndcg_results)
    return ndcg_results


def save_results(file_name,results):
    """
    Dumps the data into a specified file under predefined location.
    Arguments:
            file_name: name of the file where data will be stored
            results: data to be stored
    Returns:
            Displays either a success message or an Error message.
            In case of success, stores data in given file_name under ~data/dumps/file_name 
    
    """
    loc = 'data/dumps/'
    if not os.path.exists(loc):
        os.makedirs(loc)
    try:
        with open(loc+file_name,'wb') as f:
            pickle.dump(results,f)
    except:
        print('Error: Unable to save the results. An error encountered.')
        return
        
    print('Success: Results save successfully in file: {}'.format(loc+file_name))

def load_results(dfname,K):
    '''
    load NDCG@K results for dataframe from predefined location.
    Arguments:
            dfname: (str) data frame name. Valid values are one of ['movies','games','tv','music']
            K: (int) number of recommendations to consider
    Returns:
            Data as per the provided arguments.
            Note: the function checks for file under location '~data/dumps/'
    Exceptions:
            FileNotFound:
    
    '''
    
    loc = 'data/dumps/'
    file_name = 'ndcg_{}_{}'.format(dfname,K)
    try:
        return pickle.load(open(loc+file_name,'rb'))
    except FileNotFoundError:
        raise Exception('Error: File not found. Check for arguments provided.')
    

