from Aggregation import Agg
import numpy as np
class _model_:
    def __init__(self,train_data,train_lab,train_ind,num_cross=4,feature="",Reduction=(True,False),model=""):
        #Reduction[0]==CSP,Reduction[1]==PCA
        self.data=train_data
        self.lab=train_lab
        self.ind=train_ind
        self.num_cross=num_cross
        self.feature=feature
        self.Reduction=Reduction
        self.model=model
        self.res={}

    def cross_val(self):
        Tri=Agg()
        Tri.data=self.data
        Tri.lab=self.lab
        Tri.ind=self.ind
        Tri.create_epochs()
        cross_result=[]
        for m in range(self.num_cross):
            Tri.split(stage=m)
            if self.feature=="TimeFreq_STFT":
                Tri.STFT()
                Tri.STFT(train=False)
            if self.feature=="Freq_Welch":
                Tri.Welch()
                Tri.Welch(train=False)
            if self.feature=="MicroState":
                Tri.micro_state()
                Tri.get_micro_state()
                
            CSP=self.Reduction[0]
            PCA=self.Reduction[1]

            if CSP==True and PCA==False:
                Tri.CSP(feature=self.feature)
                Tri.CSP(train=False,feature=self.feature)
            elif CSP==False and PCA==True:
                Tri.PCA(feature=self.feature)
                Tri.PCA(train=False,feature=self.feature)

            if self.model=="LDA":
                Tri.LDA(feature=self.feature)
            if self.model=="SVM":
                Tri.SVM(feature=self.feature)
            if self.model=="RF":
                Tri.RF(feature=self.feature)
            if self.model=="GB":
                Tri.GB(feature=self.feature)

            result=Tri.predict(model=self.model,feature=self.feature)
            #cross_result.append(result[0])
            cross_result.append(result[1])
        print(cross_result)
        print("Cross_result")
        mini=min(cross_result)
        Tri.train_epoch=Tri.epoch
        Tri.train_lab=self.lab
        Tri.train_ind=self.ind
        if self.feature=="TimeFreq_STFT":
            Tri.STFT()
            map=None
        if self.feature=="Freq_Welch":
            Tri.Welch()
            map=None
        if self.feature=="MicroState":
            Tri.micro_state()
            map=Tri.map

        CSP=self.Reduction[0]
        PCA=self.Reduction[1]        
        if CSP==True and PCA==False:
            Tri.CSP(feature=self.feature)
            CSP=Tri.model["CSP"]
        elif CSP==False and PCA==True:
            Tri.PCA(feature=self.feature)
            PCA=Tri.model["PCA"]
                
        if self.model=="LDA":
            Tri.LDA(feature=self.feature)
        if self.model=="SVM":
            Tri.SVM(feature=self.feature)
        if self.model=="RF":
            Tri.RF(feature=self.feature)
        if self.model=="GB":
            Tri.GB(feature=self.feature)
        mean=np.mean(cross_result)
        var=np.var(cross_result)
        self.res.update(mean=mean,variance=var,mini=mini,feature=self.feature,Map=map,Reduction=(CSP,PCA),model=Tri.model[self.model])
        return self.res
