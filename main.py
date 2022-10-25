#%%
from Aggregation import Agg 
from Vote import vote
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
#set cross validation=4
vo=vote(train_data,train_lab,train_ind,
    feature_list=["Freq_Welch","TimeFreq_STFT"],Reduction=[(True,False),(False,True),(False,False)],model_list=["LDA","SVM"])#microstate can't use if set csp==true, need some modify
vo.compute_models()
vo.test_data=test_data
vo.test_lab=test_lab
vo.test_ind=test_ind
vo.maj_vote()


# %%
