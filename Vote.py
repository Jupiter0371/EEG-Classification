from Model import _model_
from Aggregation import Agg
import numpy as np
class vote:
    def __init__(self,train_data,train_lab,train_ind,test_data=[],test_lab=[],test_ind=[],feature_list=[],Reduction=[(True,False),(False,True),(False,False)],model_list=[]):
        self.feature_list=feature_list
        self.Reduction=Reduction
        self.model_list=model_list
        self.train_data=train_data
        self.train_lab=train_lab
        self.train_ind=train_ind
        self.test_data=test_data
        self.test_lab=test_lab
        self.test_ind=test_ind
        self.mean=[]
        self.var=[]
        self.feature=[]
        self.reduction=[]
        self.model=[]

    def compute_models(self):
        train_data=self.train_data
        train_lab=self.train_lab
        train_ind=self.train_ind
        mean=[]
        var=[]
        feature=[]
        reduction=[]
        model=[]
        for fea in self.feature_list:
            for R in self.Reduction:
                for mod in self.model_list:
                    #if C!= True or fea!="MicroState":
                    Model=_model_(train_data,train_lab,train_ind,feature=fea,Reduction=R,model=mod)
                    res=Model.cross_val()
                    mean.append(res["mean"])
                    feature.append(res["feature"])
                    var.append(res["variance"])
                    reduction.append(res["Reduction"])
                    model.append(res["model"])
        self.mean=mean
        self.var=var
        self.feature=feature
        self.reduction=reduction
        self.model=model
    def maj_vote(self):
        feature=self.feature
        reduction=self.reduction
        model=self.model
        mean=self.mean
        total_predict=[]
        index=[]
        #weight=mean/np.sum(mean)
        weight=[]
        for i in range(len(feature)):
            if mean[i]>0.5:
               index.append(i)
               weight.append(mean[i])
        weight=weight/np.sum(weight)
        #print(index)
        if len(index)==0:
            index=list(range(len(mean)))
            weight=[1/len(mean)]*len(mean)
        for j in index:
        #for j in range(len(feature)):
            Tr=Agg()
            Tr.test_data=self.test_data
            Tr.test_lab=self.test_lab
            Tr.test_ind=self.test_ind
            Tr.create_epochs(test=True)
            #get feature
            feature_test=feature[j]
            if feature_test=="Freq_Welch":
                Tr.Welch(train=False)
            elif feature_test=="TimeFreq_STFT":
                Tr.STFT(train=False)
            elif feature_test=="MicroState":
                Tr.micro_state(train=False)
            #Check whether use CSP,PCA or not
            reduction_test=reduction[j]
            CSP=reduction_test[0]
            PCA=reduction_test[1]
            if CSP!=False:
                Tr.test_Feature[feature_test]=CSP.transform(Tr.test_FeaChannel[feature_test])  
            elif PCA!=False:
                Tr.test_Feature[feature_test]=PCA.transform(Tr.test_Feature[feature_test])  
            #use model
            predict=model[j].predict(Tr.test_Feature[feature_test])
            total_predict.append(predict)
            acc=0
            for i in range(len(predict)):
                if predict[i]==self.test_lab[i]:
                    acc=acc+1
            acc=acc/len(predict)
            print(acc)
            
        final_result=[]
        print(total_predict)
        for m in range(len(total_predict[0])):
            label=[]
            for k in range(len(total_predict)):
                label.append(total_predict[k][m]*weight[k])
                #label.append(total_predict[k][m])
            mean_label=np.sum(label)
            print(mean_label)
            #mean_label=np.mean(label)
            if mean_label>=4.5:
                final_result.append(5)
            elif mean_label<4.5:
                final_result.append(4)

        #compute accuracy
        correct=0
        print(self.test_lab)
        print(final_result)
        for i in range(len(self.test_lab)):
            if self.test_lab[i]==final_result[i]:
                correct=correct+1
        acc=correct/len(self.test_lab)
        print(acc)

