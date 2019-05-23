import itertools
from collections import defaultdict
import os

import pandas as pd
import numpy as np

import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.word2vec import LineSentence
from nltk.corpus import stopwords

# Extract all the words from user_reviews summary and store as a list of words
#######################################
def extract_data(extract_from,features_to_extract = []):
    '''
    Extract the listed features from the file, save the extrated features and return the dataframe.
    ##old-Extract the user reviews summary data from dataset and store user reviews as a list of lists.
    Args:
        extract_from:[str] filename from which information will be extracted. 
                Assumes that the file is delimited by pipe '|'. Does not work for anyother delimiter type.
        save_to:[str] filename to save the extracted information. It will be stored as a csv format.
        features_to_extract:[list] list of features to extract from the extract_from filename. 
                If list is empty then all features of the dataset are extracted.
        ##old - dfname: name of dataset whose user reviews summary needs to be extracted.
        ##        Valid values are ['movies','games','music','tv']
    
    Returns:
        [pandas.DataFrame] A dataframe with extracted features. 
        ##old -A list of all the sentences/reviews found in user reviews summary.
    '''
    
    filename = os.path.join('data',extract_from)
    if len(features_to_extract):
        #try:
        df = pd.read_table(filename,delimiter = '|',engine = 'python',usecols=features_to_extract)
        #except FileNotFoundError:
            
            
        
    else:       
        df = pd.read_table(filename,delimiter = '|',engine = 'python')
    
    return df
    

def process_reviews(series):
    '''
    Proces reviews/summary to tokenize the posts and remove the english stopwords.
    
    '''
    
    
    series = series.apply(gensim.utils.simple_preprocess)
    
    eng_stopwords = stopwords.words('english')
    remove_stopwords = lambda line: [word for word in line if word not in eng_stopwords]
    
    return series.apply(remove_stopwords)


def process_features(df,list_of_prefixes):
    '''
    insert the prefixes in the dataset for user_id,item_id etc.
    '''
    sep = '_'
    columns = list(df.columns.values)
    columns.remove('summary')
    #prefix = ['u_','i_']
    
    #check if the column and prefixes lists are of same size, raise IndexError otherwise
    if len(columns) != len(list_of_prefixes):
        raise Exception('Number of features does not match the prefixes provided. \
                            features={} and list_of_prefixes={}'.format(len(columns),len(list_of_prefixes)))
    
    idx=0
    while(idx<len(columns)):
        df.iloc[:,idx:idx+1] = list_of_prefixes[idx] + sep + df.iloc[:,idx:idx+1]
        idx+=1
    
    df.summary = process_reviews(df.summary)
    
    return df  
    
def preprocess_data(filename,list_of_features=[],list_of_prefixes=[]):
    '''
    Main function, calls these functions 1. extract_data, 2. process_features
    Args: 
        filename:[str] filename of stored dataset
        list_of_features:[list] list of features to extract from the dataset
        list_of_prefixes:[list] list of prefixes to include in features other than reviews.
                Assumptions:
                    1. prefixes are provided in same order as features
                        e.g. for features=['user_id','item_id'], prefixes=['u','i']
                    2. The default seperator '_' is used
                    3. length of prefixes list is always equal to len(list_of_features)-1
    Returns:
        [pandas.DataFrame] feature columns with prefixed characters and summary column tokenized and ready for injection. 
                    
        
    '''
    if len(list_of_prefixes)!= len(list_of_features)-1:
        raise ValueError('The precondition: len(prefixes)== len(feature_lst)-1 not met. \nPlease check the arguments provided.')
    
    #1. extract features from dataset
    df = extract_data(filename,list_of_features)
    
    #clean the data to remove all null values
    df.dropna(axis=0,how='any',inplace=True)
    
    #2. process the information before insertion
    
    df = process_features(df,list_of_prefixes)
    
    return df


def inject_information_in_reviews(df,list_of_features,save_to,inject_window_size = 15):
    '''
    The function injects the information from listed features into user reviews/user posts after specified windowsize
    This function should be called after preprocessing all the data especially user posts summary.
    '''
    list_of_features.remove('summary')
    #iterate over rows of dataset
    for index,row in df.iterrows():
        #iterate over list of features and store the values
        #store values to inject in user-reviews
        val_lst = [row[val] for val in list_of_features]
        #u = row['user_id']
        #i = row['token_name']
        #injected summary
        new_summary = row['summary']
        #print(lst)
        idx = 0
        while(idx <=len(new_summary)):
            #[lst.insert(idx,a) for a in [i,u]]
            [new_summary.insert(idx,val) for val in val_lst]
            idx +=inject_window_size+len(val_lst)
        
        #replace summary with injeced summary
        row['summary'] = ' '.join(new_summary)
    
    arr = np.array(df.summary)
    
    #return arr
    #print(arr)
    filename = os.path.join('data',save_to)
    try:
        np.savetxt(filename,arr,fmt='%-5s',encoding='utf-8')
        print('File successfully saved at location: {}'.format(filename))
    except Exception as e:
        print('Error in saving injected information file.\n',e)    
    

def create_sentences(extract_from,list_of_features,list_of_prefixes,save_to,inject_window_size=15):
    '''
    This function readies the data by first reading it from the disk location and then processing the data. 
    Two processings are applied on the data:
        1. gensim.utils.simple_preprocess() removes any digits, special characters, punctuation marks from the sentences and the output is generated in list_of_list format.
        2. nltk.corpus.stopwords() is used to remove stopwords from the list_of_lists formatted output from step 1.
    Arguments:
        dfname: name of dataset whose user reviews summary needs to be extracted.
                Valid values are ['movies','games','music','tv','all']
    Returns:
        Nothing.
        Prints if the processed and injected file is saved in memory or not.
    '''
    #preprocess data
    df = preprocess_data(extract_from,list_of_features,list_of_prefixes)
    
    #inject informaion in reviews
    inject_information_in_reviews(df,list_of_features,save_to,inject_window_size)
    #read the data from disk location
    #data = read_data(dfname)
    #return exclude_stopwords(list(data))

def read_sentences(filename):
    '''
    Read the data from the disk location. 
    Using the simple_preprocess() method of gensim, 
    this method removes any words of length 2, any punctuation marks and 
    converts the posts into list_of_list [[posts]] format.
    
    Arguments:
        dfname: ['movies','games','tv','music','all'] any of these values.
    
    Assumptions:
        There are following assumptions for this method to work:
            1. Data is present at location ~/data/
            2. The file name is in the format summary_[dfname].txt
            e.g. for dfname: movies the data should be present as ~/data/summary_movies.txt
            
    '''
    #load data from file
    loc = 'data'
    #file = 'summary_{}.txt'.format(dfname)
    file_path = os.path.join(loc,filename)
    
    if os.path.isfile(file_path):
        with open(file_path,'rb') as f:
            for line in f:
                #print(line)
                yield line.split()
    else:
        raise FileNotFoundError('There is no file "{}" at location "{}"'.format(file,os.path.abspath(loc)))


#Word2Vec model Accuracy
#################################
def word_keep_rule(word, word_count, min_count):
    """
    This function is used only by gensim's word2vec. A rule to decide whether to keep a word or discard it in gensim.word2vec model.
    This rule is used to keep injected users and items in the word2vec vocabulary. We want all of these so that we can recommend things to any user or item.
    Gensim's word2vec will call this rule on every vocab word to decide if its kept. 
    If the word is prefixed by 'u:' or 'm:', we keep it. Otherwise, we choose the default behavior (which is: discard it if word_count < min_count).
    For a reference on this default behavior, see
    https://github.com/piskvorky/gensim/blob/develop/gensim/models/word2vec.py#L404
    """
    if word[:1] == 'u' or word[:1] == 'i':
        return gensim.utils.RULE_KEEP
    return gensim.utils.RULE_DEFAULT


def w2v_model_accuracy(model):
    questions = 'questions-words.txt'
    accuracy = model.wv.evaluate_word_analogies(os.path.join('data',questions))
    
    sum_corr = len(accuracy[1][-1]['correct'])
    sum_incorr = len(accuracy[1][-1]['incorrect'])
    total = sum_corr + sum_incorr
    percent = lambda a: a / total * 100
    
    #print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, percent(sum_corr), percent(sum_incorr)))
    return percent(sum_corr),percent(sum_incorr)


#Hyper-Parameterization 
####################################

def run_and_evaluate_w2v_model(sentences,vecsize,alpha,windowsize,workers,mincount,skipgram,hsoftmax,negative):
    '''
    Run and evaluate the w2v model with give set of parameters. 
    The evaluation is performed based on the question-words.txt from T.Mikolov's original accuracy document at https://github.com/tmikolov/word2vec.
    
    Arguments:
    vecsize: feature vector size
    alpha: training rate
    windowsize: stride of window 
    workers: number of workers used to train
    mincount: ignore all words with total frequeny less than this number
    skipgram: [0,1]: 1- skipgram, 0-CBOW
    hsoftmax: {0,1}: If 1, hierarchical softmax will be used for model training.
               If 0, and `negative` is non-zero, negative sampling will be used.
    negative: If > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drawn (usually between 5-20).
              If set to 0, no negative sampling is used.
    
    Libraries Dependencies:
        The method uses gensim library to compute w2v model.
    '''
    #create model
    model = gensim.models.Word2Vec(sentences,size = vecsize,alpha=alpha,window=windowsize,min_count=mincount,workers=workers,sg=skipgram,hs=hsoftmax,negative=negative)
    
    #train model
    model.train(sentences,total_examples=len(sentences),epochs=5)
    
    #calculate accuracy of model
    correct,incorrect = w2v_model_accuracy(model)
    
    return [correct,incorrect]



#Do Exhaustive grid search
#########

def run_w2v_hypertuning(sentences,param_grid):
    '''
    Run model with all parameter combinations as provided in the arguments. Make exhaustive combinations of parameters.
    Arguments:
        data: Ready to train data in the form of [[sentences]]
        param_grid: [a dict object] a dictionary with parameters to change. Currenty only excepting four parameters criterion,max_depth,min_sampels_leaf,max_features
    Returns:
        A dictionary object with parameter combinations and their accuracy results. 
    Limitations:
        Currently, this function is limited to:
        1. Word2Vec model using gensim library and does not apply to anyother model. Modifications may be required to run for other models.
        2. For W2V parameters considered for training are size, alpha, window, workers, min_count, sg, negative. Other parameters are not used for training the model.
    
    Libraries dependency:
        This method uses these libraries: itertools, defaultdict.
    '''
    #include required libraries
    #import itertools
    #from collections import defaultdict
    
    try:
        size = param_grid['size']
        alpha = param_grid['alpha']
        window = param_grid['window']
        workers = param_grid['workers']
        min_count = param_grid['min_count']
        skipgram = param_grid['sg']
        hsoftmax = param_grid['hs']
        negative = param_grid['negative']
    except KeyError as e:
        raise KeyError('Required parameters not provided in param_grid. Make sure these parameters are included: ',e)
        
    
    result_dict = defaultdict(list)
    
    for param_set in itertools.product(size, alpha, window, workers, min_count, skipgram,hsoftmax, negative):
        sz = param_set[0]
        a = param_set[1]
        wndw = param_set[2]
        wrkr = param_set[3]
        mincnt = param_set[4]
        sg= param_set[5]
        hs = param_set[6]
        ngtv = param_set[7]
        
        
        # run and evaluate w2v model
        acc_result = run_and_evaluate_w2v_model(sentences,sz,a,wndw,wrkr,mincnt,sg,hs,ngtv)

        #store parameters in results_dict
        result_dict['size'].append(sz)
        result_dict['alpha'].append(a)
        result_dict['window'].append(wndw)
        result_dict['workers'].append(wrkr)
        result_dict['min_count'].append(mincnt)
        result_dict['sg'].append(sg)
        result_dict['hs'].append(hs)
        result_dict['negative'].append(ngtv)
        #store model evaluation results in results dictionary
        result_dict['correct_per'].append(acc_result[0])
        result_dict['incorrect_per'].append(acc_result[1])
    
    #store results as a pandas Dataframe in pickle
    pd.DataFrame(result_dict).to_pickle(os.path.join('data','w2v_grid_search_results.pkl'))
    #return result dictionary    
    return result_dict