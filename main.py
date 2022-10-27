#%%
from Aggregation import Agg 
from Vote import vote
from Model import _model_ 
#split the offline data into train and test set
Ori=Agg()#
Ori.read_private_data(path="//XTZJ-20220530VW/shared/课题合作/南方医科大学南方医院白云分院/TMS-EEG/eeg_data/LZH000464689/2022.09.15-15.52.21-QEEG-LZH000464689.edf")
Ori.raw_to_data()
Ori.split_train_test()
train_data=Ori.train_data
train_lab=Ori.train_lab
train_ind=Ori.train_ind
test_data=Ori.test_data
test_lab=Ori.test_lab
test_ind=Ori.test_ind#Reduce for real case

#compute the models
vo=vote(train_data,train_lab,train_ind,
    feature_list=["MicroState","TimeFreq_STFT","Freq_Welch"],Reduction=[(True,False),(False,True),(False,False)],model_list=["LDA","SVM"])#(0,1) 0: CSP; 1: PCA
vo.compute_models()

#compute the final result
vo.test_data=test_data
vo.test_lab=test_lab
vo.test_ind=test_ind
vo.maj_vote()






# %%
