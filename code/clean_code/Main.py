
# coding: utf-8

# In[3]:


import recommendations as rec
import pandas as pd


# In[5]:


#k_list = [2,4,8,16,32,64]
k_list = [4]
results = pd.DataFrame(data = None,columns = [['k','rmse_cv','rmse_test','file','method']])
loc=0

method_list = ['ubcf','ibcf']
#for file in ['tv','music','movies','games']:
for file in ['games']:
    train,test = rec.load_train_test_data(file,test_fold=1,nfolds=5)
        
    for method in method_list:
        print('*********Started working on database:{} and collaborative filtering: {} '.format(file,method))
        print('*'*40)
        cf = True
        if method =='ibcf':
            cf = False
            
        for k in k_list:
            print('........Calculating Results for K={}.........'.format(k))
            
            rmse_cv,rmse_test = rec.knn_model_performance(train,test,k,user_based =cf,dfname=file)
            
            results.loc[loc]=k,rmse_cv,rmse_test,file,method
            loc+=1
    
    rec.save_rmse_results('rmse_results_{}'.format(file),results)
    

