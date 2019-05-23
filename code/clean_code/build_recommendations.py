
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os,json

from surprise import accuracy
from surprise import Reader,Dataset,KNNBasic,model_selection


def load_train_test_data(file_name,test_fold,nfolds=5):
    """
    Loads the data from csv files to train-test dataframes as per the arguments.
    Arguments:
        file_name: file to load the data from. Should not contain the fold number or extension from original file.
        test_fold: an int. This specifies the file number that should be considered as test data.
        nfolds: an int specifying the number of folds of original file on disk.
    Returns:
        Two pandas dataframes.
        Train: Contains data from all fold files except test_fold.
        Test: Contains data only from specified test_fold file.
    
    """
    folds = list(range(1,nfolds+1))
    #test_fold = 1
    test = pd.DataFrame()
    train = pd.DataFrame()
    for fold in folds:

        if fold==test_fold:
            test = pd.read_csv('data//{}{}.csv'.format(file_name,fold),delimiter='|')
        else:
            d = pd.read_csv('data//{}{}.csv'.format(file_name,fold),delimiter='|',index_col=0)
            train = pd.concat([train,d],axis = 0)
    train.reset_index(inplace=True)
    return train,test


def knn_model_performance(train,test,k,user_based=True,nfolds=5,dfname = 'tv'):
    '''
    Function to calculate KNNBasic model performance on train and test datasets. The performance measure used is RMSE. 
    The train set RMSE is calculated based on n-fold cross validation.
    Arguments: 
        train: training dataset object
        test: testing dataset object
        k: (int) number of neighbors for KNNBasic model
        user_based: (binary)True for user-based collaborative filtering and False for Item-based collaborative filtering
                    Default value is True i.e. user-based cf
        nfolds: (int) number of folds for cross validation.
                Default value is 5-fold cross validation
        dfname: valid values are ['movies','games','tv','music']
                Default value is 'tv'
    Returns:
        RMSE result for cross-validation and RMSE result for test set.
        
    '''
    reader = Reader(line_format='user item rating',rating_scale = (1,10))
    train_data = Dataset.load_from_df(train,reader = reader)
    test_data = Dataset.load_from_df(test,reader = reader)
    #define knn algorithm to use
    #if user_based:
    algo = KNNBasic(k=k,sim_options={'user_based':user_based})
    #else:
    #algo = KNNBasic(k=k,sim_options={'user_based':False})
    
    cv_results = model_selection.cross_validate(algo,train_data,                                                measures=['rmse'],cv=nfolds,n_jobs = -1,pre_dispatch=1)
    #cv_results = model_selection.cross_validate(algo,train_data,\
    #                                            measures=['rmse'],cv=nfolds)
    
    rmse_cv = sum(cv_results['test_rmse'])/nfolds
    print('RMSE Cross Validation Tests: ',cv_results['test_rmse'])
    print('Mean RMSE Cross Validation Tests: ',rmse_cv)
    
    #fit the model on full training data
    trainset = train_data.build_full_trainset()
    algo.fit(trainset)
    
    #run the model on validation testset to get predictions
    ts = test_data.build_full_trainset()
    testset = ts.build_anti_testset()
    predictions = algo.test(testset)
    if user_based:
        filtering = 'ubcf'
    else:
        filtering = 'ibcf'
    #save prediction results on disk
    file_name = '{}_{}_fold1_knn{}'.format(filtering,dfname,k)
    save_results(file_name,predictions)
    #save_results_games(file_name,predictions)
    rmse_test = accuracy.rmse(predictions)
    print('RMSE of test set predictions: ',rmse_test)
    return rmse_cv, rmse_test

def save_results(file_name,results):
    '''
    Dump the results in json file. The results are stored in the order:
        user_id,item_id,actual_rating(global mean, if actual rating not known),predicted_rating, other info
        
    Arguments:
            file_name:(str) File name with path where file will be dumped
            results: The list of predictions to store
            
    Output:
        The resulting file will be stored at location '~/data/dumps/file_name'
        
    '''
    
    loc = 'data/dumps/'
    if not os.path.exists(loc):
        os.makedirs(loc)
    
    with open(loc+file_name,'w') as f:
        json.dump(results,f)
    
    return print('Results saved successfully.')
        
def load_results(filter_method,dfname,k=2):
    """
    Load prediction results from the saved files on disk location.
    Arguments:
        filter_method: either of [ubcf,ibcf]
        dfname: dataset name, the dataset that needs uploading.
                Valid values ['movies','games','tv','music']
        k: number of nearest neighbors
    Returns:
        Saved Prediction results for the specified file. The results are returned in below format:
        user_id,item_id,actual_rating(global mean in case actual not present),estimated_rating,other_info
    """
    loc = 'data/dumps'
    file_name = loc+'/{}_{}_fold1_knn{}.json'.format(filter_method,dfname,k)
    
    return json.load(open(file_name,'r'))


def get_top_n_predictions(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

