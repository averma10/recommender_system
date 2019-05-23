
import pandas as pd

import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.word2vec import LineSentence
from nltk.corpus import stopwords

import itertools
from collections import defaultdict
import os


# Extract all the words from user_reviews summary and store as a list of words
#######################################

def build_data(dfname):
    '''
    Extract the user reviews summary data from dataset and store user reviews as a list of lists.
    Args:
        dfname: name of dataset whose user reviews summary needs to be extracted.
                Valid values are ['movies','games','music','tv']
    Returns:
        A list of all the sentences/reviews found in user reviews summary.
    '''
    filename = os.path.join('data','user_reviews_by_{}.csv').format(dfname)
    df = pd.read_csv(filename,delimiter = '|',engine = 'python')
    
    arr = []
    filename = os.path.join('data','summary_{}.txt').format(dfname)
    
    arr = np.array(df.summary)
    
    np.savetxt(filename,arr,fmt='%-5s',encoding='utf-8')
    
    print('Summary Data successfully saved in file:"{}"'.format(filename))
    
    
def exclude_stopwords(list_of_sentences):
    '''
    Function to remove stopwords from sentences. This function uses nltk library and removes only english stopwords.
    Use this function after splitting the sentences into word list
    Requirements:
        This function requires the stopwords data downloaded on local machine. Before calling this function ensure to download the stopwords using nltk.download('stopwords')
    Arguments:
        list_of_sentences:[list of lists] list of sentences where each sentence is a list of words.
    Returns:
        [list of lists] After removing stopwords: list of sentences where each sentence is a list of words.
    '''
    try:
        stopwords_set = set(stopwords.words('english'))
    except LookupError as e:
        raise Exception('Not able to find stopwords. Check if these are downloaded on local machine.\nError Message:\n',e)
    
    return [[word for word in sentence if word not in stopwords_set] \
             for sentence in list_of_sentences]


def read_data(dfname):
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
    file = 'summary_{}.txt'.format(dfname)
    file_path = os.path.join(loc,file)
    
    if os.path.isfile(file_path):
        with open(file_path,'rb') as f:
            for line in f:
                yield gensim.utils.simple_preprocess(line)
    else:
        raise FileNotFoundError('There is no file "{}" at location "{}"'.format(file,os.path.abspath(loc)))

        
        
        
def ready_data(dfname = 'all'):
    '''
    This function readies the data by first reading it from the disk location and then processing the data. 
    Two processings are applied on the data:
        1. gensim.utils.simple_preprocess() removes any digits, special characters, punctuation marks from the sentences and the output is generated in list_of_list format.
        2. nltk.corpus.stopwords() is used to remove stopwords from the list_of_lists formatted output from step 1.
    Arguments:
        dfname: name of dataset whose user reviews summary needs to be extracted.
                Valid values are ['movies','games','music','tv','all']
    Returns:
        [[lists]] Processed data ready for use in Word2Vec model
    '''
    #read the data from disk location
    data = read_data(dfname)
    return exclude_stopwords(list(data))


#Word2Vec model Accuracy
#################################

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