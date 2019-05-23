
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_rmse(results,dfname='movies',rmse = 'test'):
    '''
    function to plot RMSE results of recommendations on different datasets
    Arguments:
            results: (pandas.DataFrame()) dataframe with RMSE results of predictions
            dfname: one of ['movies','games','tv','music']
                    Default value is 'movies'
            rmse: valid values ['test','train']. 
                    Default value is 'test'
    Return:
            Figure with RMSE plot
    
    '''
    k = list(results[results.method=='ubcf']['k'])
    if rmse == 'test':
        ubcf = results[results.method=='ubcf']['rmse_test']
        ibcf = results[results.method=='ibcf']['rmse_test']
        title = 'RMSE results for {} Test Set'.format(dfname.capitalize())
    else:
        ubcf = results[results.method=='ubcf']['rmse_cv']
        ibcf = results[results.method=='ibcf']['rmse_cv']
        title = 'RMSE results for {} 5-fold Cross Validation'.format(dfname.capitalize())
    
    p1, = plt.plot(k,ubcf,'r',label='ubcf')
    p2, = plt.plot(k,ubcf,'ro')
    p3, = plt.plot(k,ibcf,'b',label='ibcf')
    p4, = plt.plot(k,ibcf,'bo')
    plt.xlabel('Value of K for KNN')
    plt.ylabel('RMSE')
    plt.xticks(k)
    plt.title(title)
    #plt.ylim(1.8,2.4)
    plt.legend(handles = [p1,p3])
    

def plot_rmse_all_results():
    #load rmse results for all datasets
    #####################
    ###Write Code here####
    ###############
    #plot results
    plt.figure(figsize=(13,10))

    plt.subplot(221)
    plot_rmse(results_movies,'movies')
    plt.ylim(1.6,2.4)

    plt.subplot(222)
    plot_rmse(results_music,'music')
    plt.ylim(1.6,2.4)

    plt.subplot(223)
    plot_rmse(results_tv,'tv')
    plt.ylim(1.6,2.4)

    plt.subplot(224)
    plot_rmse(results_tv,'tv')
    plt.ylim(1.6,2.4)

    plt.show()


###########################################
####  Might Need to Change this for IBCF ##
###########################################
def get_all_relevance(predictions,user_based = True):
    '''
    returns a list of prediction ratings related to user_id/item_id as returned by our KNNBasic model
    Arguments:
        predictions: all the test predictions returned by model.
        user_based: (binary) True will create dictionary with user_id, ratings pair and False will create dictionary with item_id, ratings pair.
        
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


def plot_ndcg(results, dfname='movies', K=20):
    '''
    function to plot NDCG@K metric for recommendations
    Arguments:
            results: (pandas.DataFrame()) dataframe with NDCG results of predictions
            dfname: one of ['movies','games','tv','music']
                    Default value is 'movies'
            K: (int) number of recommendations to consider
                    Default value is 20
    Return:
            Figure with NDCG plot
    
    '''
    
    k = list(results[results.method=='ubcf']['k'])
    ubcf = results[results.method=='ubcf']['ndcg']
    ibcf = results[results.method=='ibcf']['ndcg']
    title = 'NDCG Results for {} Predictions at different k-values'.format(dfname.capitalize())
    
    p1, = plt.plot(k,ubcf,'r',label='ubcf')
    p2, = plt.plot(k,ubcf,'ro')
    p3, = plt.plot(k,ibcf,'b',label='ibcf')
    p4, = plt.plot(k,ibcf,'bo')
    plt.xlabel('Value of K for KNN')
    plt.ylabel('NDCG@{}'.format(K))
    plt.xticks(k)
    plt.title(title)
    #plt.ylim(1.8,2.4)
    plt.legend(handles = [p1,p3])
    
def get_ndcg_results(dfname,K=20):
    '''
    Compute NDCG@K results and store on disk location for later reference.
    Arguments:
        dfname: data frame name. Valid values ['movies','games','tv','music']
        K: (int) number of recommendations to consider
    
    Return:
        NDCG@K
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

def load_and_plot_ndcg(dfname,k):
    '''
    function to load and plot NDCG@K results conveniently
    Arguments:
            dfname: dataframe name ['movies','games','tv','music']
            k: (int) number of recommendations to consider
    Return:
            Figure with plot of NDCG@K
    '''
    loc = 'data/dumps'
    fn = 'ndcg_{}_{}'.format(dfname,k)
    #load the dataframe
    #might need to change if I proceed with json format
    df = load(loc + '/' + fn)[0]
    
    plot_ndcg(df,dfname,k)
    
def plot_ndcg_all_results(plot_function,no_of_items=50):
    '''
    Plot NDCG@K plots in one figure for all the data frames.
    Arguments:
            plot_function: function to call for ploting
            no_of_items: (int) number of recommendations to consider
    Returns:
        NDCG@K plots for all data frames in one figure.
    
    '''
    plt.figure(figsize=(12,10))
    plt.subplot(221)
    plot_function('movies',no_of_items)
    plt.ylim(0.70,0.90)
    
    plt.subplot(222)
    #plot_function('games',no_of_items)
    plot_function('movies',no_of_items)
    plt.ylim(0.70,0.90)
    
    plt.subplot(223)
    plot_function('music',no_of_items)
    plt.ylim(0.70,0.90)
    
    plt.subplot(224)
    plot_function('tv',no_of_items)
    plt.ylim(0.70,0.90)
    
    plt.show()

