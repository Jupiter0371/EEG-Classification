import mne
import numpy as np
import scipy
import mne_microstates
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
#from eegPro import eegPro
import os
import scipy
import scipy.io as scio
from scipy.linalg import sqrtm,inv,pinv
from mne.preprocessing import ICA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV    
from mne.decoding import CSP,EMS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from mne_connectivity import Connectivity
import matplotlib.pyplot as plt

class Agg:
    def __init__(self,data=[],event_lab=[],event_ind=[],freq=500,tmin=-1,tmax=3):
        '''data:19通道的data
           event_lab: event 对应的label
           event_ind: event 对应的index
           freq: 采样率
           tmin：数据截取的开始时刻（相对发生event的时间）
           tmax：数据截取的截止时刻（相对发生event的时间）
        '''
        self.data=data
        self.ind=event_ind
        self.lab=event_lab
        self.features=[]
        self.epoch=[]
        self.raw=[]
        self.t_interval=0
        self.freq=freq
        self.train_data=[]
        self.test_data=[]
        self.train_lab=[]
        self.test_lab=[]
        self.train_ind=[]
        self.test_ind=[]
        self.train_epoch=[]
        self.test_epoch=[]
        self.Feature={}
        self.train_Feature={}#Feature without Channel, one-dim for each event, can do classification
        self.train_FeaChannel={}#Feature with Channel(Can do CSP)
        self.test_FeaChannel={}
        self.test_Feature={}
        self.map=[]
        self.model={}
        self.tmin=tmin
        self.tmax=tmax



#####################################play with data#############################################################################################################

    def read_private_data(self,path):#+++++判断path是否格式正确
        '''
        Args:
            filepath: end by .edf
        return:
            raw_data, lab, ind of lab
        '''
        filepath=path
        raw=mne.io.read_raw_edf(filepath,preload=True)
        raw.drop_channels(['X1', 'X2', 'LABEL', 'Ch1', 'Ch2', 'VideoSensor', 'Audio', 'MouseButton'])
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage)
        event_mapping={"FRAG:31":4, "FRAG:32":5}
        (event_from_annot, event_dic)=mne.events_from_annotations(raw,event_id=event_mapping)
        event_lab=[]
        event_ind=[]
        for i in range(len(event_from_annot)):
            event_lab.append(event_from_annot[i][2])
            event_ind.append(event_from_annot[i][0])
        epoch_lab = np.array(event_lab)
        event_ind=np.array(event_ind)
        data=raw.get_data()
        self.raw=raw
        self.ind=event_ind
        self.lab=epoch_lab

    def raw_to_data(self):
        '''
        Args:
            None
        return:
            19-channel*time dataset
        '''
        raw=self.raw
        self.data=raw.get_data()

    def find_opttime(self,nperseg=128,low_freq=7,high_freq=30):
        '''
        Args:
            nperseg: Length of each segment
            low_freq: low frequency we want
            high_freq:high frequency we want 
        return:
            the optimal time point we use to create epochs
            (equal to the mean of C3-channel and C4-channel of the largest difference between left and right MI)
        '''

        data=self.train_data 
        ind=self.train_ind
        lab=self.train_lab 
        tmin=self.tmin 
        tmax=self.tmax 
        freq=self.freq
        ori_right_C3_epoch=[]
        ori_right_C4_epoch=[]
        ori_left_C3_epoch=[]
        ori_left_C4_epoch=[]
        #提取其中的C3，C4channel，并分别做傅里叶变换, C3:No9,C4 No11
        for i in range(len(ind)):
            if lab[i]==4:#左
                epoch1=data[8,ind[i]+tmin*freq:ind[i]+tmax*freq]
                epoch2=data[10,ind[i]+tmin*freq:ind[i]+tmax*freq]
                ori_left_C3_epoch.append(epoch1)
                ori_left_C4_epoch.append(epoch2)
            if lab[i]==5:
                epoch1=data[8,ind[i]+tmin*freq:ind[i]+tmax*freq]
                epoch2=data[10,ind[i]+tmin*freq:ind[i]+tmax*freq]
                ori_right_C3_epoch.append(epoch1)
                ori_right_C4_epoch.append(epoch2)

        trend_C3_L=[]
        for l in ori_left_C3_epoch:
            f,t,Zxx=scipy.signal.stft(l,fs=freq,nperseg=nperseg)
            Zxx=np.abs(Zxx)
            for i in range(len(f)):
                if f[i]>=low_freq:
                    low=i 
                    break
            for j in range(len(f)):
                if f[j]>=high_freq:
                    high=j
                    break 
            for t_l in range(len(t)):
                if t[t_l]>=1:
                    low_t=t_l 
                    break
            Zxx_base=Zxx[low:high+1,:low_t]
            Zxx_remain=Zxx[low:high+1,low_t:]
            Zxx_mean=np.mean(Zxx_base)
            ERDS_C3_left=[]
            for t in range(len(Zxx_remain[0])):
                i=Zxx_remain[:,t]
                i_mean=np.mean(i)
                ERDS_percent=(i_mean-Zxx_mean)/Zxx_mean
                ERDS_C3_left.append(ERDS_percent)
            trend_C3_L.append(ERDS_C3_left)
        ERDS_C3_left=np.array(trend_C3_L)
        trend_C3_left=np.mean(ERDS_C3_left,axis=0)

        trend_C3_R=[]
        for l in ori_right_C3_epoch:
            f,t,Zxx=scipy.signal.stft(l,fs=freq,nperseg=nperseg)
            Zxx=np.abs(Zxx)
            for i in range(len(f)):
                if f[i]>=low_freq:
                    low=i 
                    break
            for j in range(len(f)):
                if f[j]>=high_freq:
                    high=j
                    break 
            for t_l in range(len(t)):
                if t[t_l]>=1:
                    low_t=t_l 
                    break

            Zxx_base=Zxx[low:high+1,:low_t]
            Zxx_remain=Zxx[low:high+1,low_t:]
            Zxx_mean=np.mean(Zxx_base)
            ERDS_C3_right=[]
            for t in range(len(Zxx_remain[0])):
                i=Zxx_remain[:,t]
                i_mean=np.mean(i)
                ERDS_percent=(i_mean-Zxx_mean)/Zxx_mean
                ERDS_C3_right.append(ERDS_percent)
            trend_C3_R.append(ERDS_C3_right)
        ERDS_C3_right=np.array(trend_C3_R)
        trend_C3_right=np.mean(ERDS_C3_right,axis=0)


        trend_C4_R=[]
        for l in ori_right_C4_epoch:
            f,t,Zxx=scipy.signal.stft(l,fs=freq,nperseg=nperseg)
            Zxx=np.abs(Zxx)
            for i in range(len(f)):
                if f[i]>=low_freq:
                    low=i 
                    break
            for j in range(len(f)):
                if f[j]>=high_freq:
                    high=j
                    break 
            for t_l in range(len(t)):
                if t[t_l]>=1:
                    low_t=t_l 
                    break
            Zxx_base=Zxx[low:high+1,:low_t]
            Zxx_remain=Zxx[low:high+1,low_t:]
            Zxx_mean=np.mean(Zxx_base)
            ERDS_C4_right=[]
            for t in range(len(Zxx_remain[0])):
                i=Zxx_remain[:,t]
                i_mean=np.mean(i)
                ERDS_percent=(i_mean-Zxx_mean)/Zxx_mean
                ERDS_C4_right.append(ERDS_percent)
            trend_C4_R.append(ERDS_C4_right)
        ERDS_C4_right=np.array(trend_C4_R)
        trend_C4_right=np.mean(ERDS_C4_right,axis=0)


        trend_C4_L=[]
        for l in ori_left_C4_epoch:
            f,t,Zxx=scipy.signal.stft(l,fs=freq,nperseg=nperseg)
            time_list=t
            Zxx=np.abs(Zxx)
            for i in range(len(f)):
                if f[i]>=low_freq:
                    low=i 
                    break
            for j in range(len(f)):
                if f[j]>=high_freq:
                    high=j
                    break 
            for t_l in range(len(t)):
                if t[t_l]>=1:
                    low_t=t_l 
                    break
            Zxx_base=Zxx[low:high+1,:low_t]
            Zxx_remain=Zxx[low:high+1,low_t:]
            Zxx_mean=np.mean(Zxx_base)
            ERDS_C4_left=[]
            for t in range(len(Zxx_remain[0])):
                i=Zxx_remain[:,t]
                i_mean=np.mean(i)
                ERDS_percent=(i_mean-Zxx_mean)/Zxx_mean
                ERDS_C4_left.append(ERDS_percent)
            trend_C4_L.append(ERDS_C4_left)
        ERDS_C4_left=np.array(trend_C4_L)
        trend_C4_left=np.mean(ERDS_C4_left,axis=0)

        x_axis=time_list[low_t:]

        plt.figure(1)
        plt.subplot(211)
        plt.plot(x_axis,trend_C3_left,"r")
        plt.subplot(211)
        plt.plot(x_axis,trend_C3_right)
        plt.subplot(212)
        plt.plot(x_axis,trend_C4_left,"r")
        plt.subplot(212)
        plt.plot(x_axis,trend_C4_right)
        plt.show()

    def split_train_test(self,train_ratio=0.8):
        ratio=train_ratio
        lab=self.lab
        ind=self.ind
        data=self.data 

        self.train_data=data[:,:ind[int(len(ind)*ratio)]]
        self.train_lab=lab[:int(len(lab)*ratio)]
        self.train_ind=ind[:int(len(ind)*ratio)]
        self.test_data=data[:,ind[int(len(ind)*ratio)-6]:]
        self.test_lab=lab[int(len(lab)*ratio):]
        test_ind=ind[int(len(ind)*ratio):]
        length=ind[int(len(ind)*ratio)-6]-1
        self.test_ind=[x-length for x in test_ind]



    def create_epochs(self,tmin=-1,tmax=3,test=False):
        '''
        Args:
            tmin: the starting time of epoch(relative to the stimulation)
            tmax: the ending time of epoch(relative to the stimulation)
            test: whether use to create epoch of the training dataset or testing dataset
        return:
            self.epoch: the epoch based on the training dataset
            self.test_epoch: the epoch based on the testing dataset
        '''
        if tmin>=tmax:
            raise ValueError("the minimum time must be greater than the maximum time.")
        if test==False:
            data=self.data
            ind=self.ind
            freq=self.freq     
            epochs=[]
            for i in range(len(ind)):
                epoch=data[:,int(ind[i]+tmin*freq):int(ind[i]+tmax*freq)]
                epochs.append(epoch)
            if len(epochs[0][0])!=len(epochs[-1][0]):#check 最后一个是否长度和之前的一样
                epochs.pop()
                lab=self.lab.tolist()
                lab.pop()
                self.lab=np.array(lab)
            self.epoch=np.array(epochs)
        elif test==True:
            data=self.test_data
            ind=self.test_ind
            freq=self.freq     
            epochs=[]
            for i in range(len(ind)):
                epoch=data[:,int(ind[i]+tmin*freq):int(ind[i]+tmax*freq)]
                epochs.append(epoch)
            if len(epochs[0][0])!=len(epochs[-1][0]):#check 最后一个是否长度和之前的一样
                epochs.pop()
                lab=self.lab.tolist()
                lab.pop()
                self.test_lab=np.array(lab)
            self.test_epoch=np.array(epochs)




            
    def split(self,cross_time=4,stage=0):
        '''
        Args:
            cross_time: the number of time we use to do cross validation
            stage: the test part in the cross validation stage 
        return:
            self.train_lab: labels of training part of the cross-validation stage
            self.train_epoch: epochs of training part of the cross validation stage
            self.test_lab:labels of testing part of the cross-validation stage
            self.test_epoch: epochs of testing part of the cross-validation
        '''
        if stage>=cross_time:
            raise ValueError("Stage must be less than the number of cross time we set")

        ratio=(cross_time-1)/cross_time
        lab=self.lab
        epoch=self.epoch
        
        #Split the data based on the ratio
        period=1-ratio
        total_period=int(1//period)
        lab_list=[]
        epoch_list=[]
        for i in range(1,total_period+1,1):
            label=lab[int(len(lab)*period)*(i-1):int(len(lab)*period)*i]
            epo=epoch[int(len(lab)*period)*(i-1):int(len(lab)*period)*i,:,:]
            lab_list.append(label)
            epoch_list.append(epo)
        remain=lab[int(len(lab)*ratio)*total_period+1:]
        if len(remain)!=0:
            lab_re=lab[int(len(lab)*ratio)*total_period+1:]
            epo_re=epoch[int(len(lab)*ratio)*total_period+1:,:,:]
            lab_list.append(lab_re)
            epoch_list.append(epo_re)
        
        train_lab=[]
        train_epoch=[]
        for i in range(len(lab_list)):
            if i != stage:
                for j in range(len(lab_list[i])):
                    train_lab.append(lab_list[i][j])
                    train_epoch.append(epoch_list[i][j])

        self.train_lab=np.array(train_lab)
        self.train_epoch=np.array(train_epoch)
        self.test_lab=np.array(lab_list[stage])
        self.test_epoch=np.array(epoch_list[stage])
        #except ValueError:
        #    print("stage is out of rank") 


    
    
    
#######################################feature extraction######################################################################################################333
    def Welch(self,train=True,low_freq=0,high_freq=200):
        '''
        Args:
            low_freq: the lowest frequency we use to extract data from the frequency domain
            high_freq: the highest frequency we use to extract data from the frequency domain
            train: wether this function deals with training data or testing data
        return:
            self.train_FeaChannel["Freq_Welch"]: train_data after transformation 19*frequency form
            self.train_Feature["Freq_Welch"]: train_data after transformation [channel1_freq, channel2_freq,... channel19_freq]
            self.test_FeaChannel["Freq_Welch"]: test_data after transformation 19*frequency form
            self.test_Feature["Freq_Welch"]: test_data after transformation [channel1_freq, channel2_freq,... channel19_freq]

        '''
        if high_freq>=self.freq:
            raise ValueError("the highest frequency must be lower than the sampling frequency")
        if low_freq<0:
            raise ValueError("the lowest frequency must be greater than zero")
        if train==True:
            epoch=self.train_epoch
            freq=self.freq
            #feature=self.epoch_withFeature
            tmp=[]
            welch=[]
            for i in range(len(epoch)):
                each_epoch=epoch[i,:,:]
                f,Pxx_den=scipy.signal.welch(each_epoch,freq)##频率分段f[0:25]对应0到45
                for i in range(len(f)):
                    if f[i]>=low_freq:
                        low=i
                        break
                for j in range(len(f)):
                    if f[j]>=high_freq:
                        high=j
                        break
                Pxx=Pxx_den[:,low:high+1]
                tmp.append(Pxx)
            tmp=np.array(tmp)
            tmp=tmp.astype(float)
            self.train_FeaChannel.update(Freq_Welch=tmp)

            for i in range(len(epoch)):
                Tmp=[]
                m=tmp[i].tolist()
                for j in range(len(Pxx)):
                    for k in range(len(Pxx[0])):
                        Tmp.append(m[j][k])
                welch.append(Tmp)
            welch=np.array(welch)
            welch=welch.astype(float)
            self.train_Feature.update(Freq_Welch=welch)

        if train==False:
            epoch=self.test_epoch
            freq=self.freq
            #feature=self.epoch_withFeature
            tmp=[]
            welch=[]
            for i in range(len(epoch)):
                each_epoch=epoch[i,:,:]
                f,Pxx_den=scipy.signal.welch(each_epoch,freq)
                for i in range(len(f)):
                    if f[i]>=low_freq:
                        low=i
                        break
                for j in range(len(f)):
                    if f[j]>=high_freq:
                        high=j
                        break
                Pxx=Pxx_den[:,low:high+1]
                tmp.append(Pxx)
            tmp=np.array(tmp)
            tmp=tmp.astype(float)
            self.test_FeaChannel.update(Freq_Welch=tmp)

            for i in range(len(epoch)):
                Tmp=[]
                m=tmp[i].tolist()
                for j in range(len(Pxx)):
                    for k in range(len(Pxx[0])):
                        Tmp.append(m[j][k])
                welch.append(Tmp)
            welch=np.array(welch)
            welch=welch.astype(float)
            self.test_Feature.update(Freq_Welch=welch)



    def STFT(self,nperseg=128,train=True,low_freq=0,high_freq=200):
        #############33STFT name
        '''
        Args:
            low_freq: the lowest frequency we use to extract data from the frequency domain
            high_freq: the highest frequency we use to extract data from the frequency domain
            train: wether this function deals with training data or testing data
        return:
            self.train_FeaChannel["Freq_Welch"]: train_data after transformation 19*frequency form
            self.train_Feature["Freq_Welch"]: train_data after transformation [channel1_freq, channel2_freq,... channel19_freq]
            self.test_FeaChannel["Freq_Welch"]: test_data after transformation 19*frequency form
            self.test_Feature["Freq_Welch"]: test_data after transformation [channel1_freq, channel2_freq,... channel19_freq]

        '''
        if high_freq>=self.freq:
            raise ValueError("the highest frequency must be lower than the sampling frequency")
        if low_freq<0:
            raise ValueError("the lowest frequency must be greater than zero")
        
        if train==True:
            epoch=self.train_epoch
            freq=self.freq
            nperseg=nperseg
            tmp=[]
            for i in range(len(epoch)):
                a=[]
                each_epoch=epoch[i,:,:]
                f,t,Zxx=scipy.signal.stft(each_epoch,fs=freq,nperseg=nperseg)
                for i in range(len(f)):
                    if f[i]>=low_freq:
                        low=i
                        break
                for j in range(len(f)):
                    if f[j]>=high_freq:
                        high=j
                        break
                Zxx=Zxx[:,low:high+1,:]
                for k in range(len(Zxx)):
                    Tmp=[]
                    for m in range(len(Zxx[0])):
                        for n in range(len(Zxx[0][0])):
                            Tmp.append(Zxx[k,m,n])
                    a.append(Tmp)
                tmp.append(a)
            tmp=np.array(tmp)
            tmp=tmp.astype(float)
            self.train_FeaChannel.update(TimeFreq_STFT=tmp)
            stft=[]
            for i in range(len(epoch)):
                m=tmp[i]
                Tmp=[]
                for j in range(len(m)):
                    for k in range(len(m[0])):
                        Tmp.append(m[j][k])
                stft.append(Tmp)
            stft=np.array(stft)
            stft=stft.astype(float)
            self.train_Feature.update(TimeFreq_STFT=stft)

        if train==False:
            epoch=self.test_epoch
            freq=self.freq
            nperseg=nperseg
            tmp=[]
            for i in range(len(epoch)):
                a=[]
                each_epoch=epoch[i,:,:]
                f,t,Zxx=scipy.signal.stft(each_epoch,fs=freq,nperseg=nperseg)
                for i in range(len(f)):
                    if f[i]>=low_freq:
                        low=i
                        break
                for j in range(len(f)):
                    if f[j]>=high_freq:
                        high=j
                        break
                Zxx=Zxx[:,low:high+1,:]
                for k in range(len(Zxx)):
                    Tmp=[]
                    for m in range(len(Zxx[0])):
                        for n in range(len(Zxx[0][0])):
                            Tmp.append(Zxx[k,m,n])
                    a.append(Tmp)
                tmp.append(a)
            tmp=np.array(tmp)
            tmp=tmp.astype(float)
            self.test_FeaChannel.update(TimeFreq_STFT=tmp)
            stft=[]
            for i in range(len(epoch)):
                m=tmp[i]
                Tmp=[]
                for k in range(len(m)):
                    for r in range(len(m[0])):
                        Tmp.append(m[k][r])
                stft.append(Tmp)
            stft=np.array(stft)
            stft=stft.astype(float)
            self.test_Feature.update(TimeFreq_STFT=stft)


    def micro_state(self,num_states=4):
        '''
        Args:
            num_states: number of cluster we use to do the data clustering(traditionally 4 is optimal)
        return:
            self.train_Feature["MicroState"]: train_data after transformation (one-dimension, containing all info in 19-channel)
            self.map: the center of each clustering
        '''
        train_epoch=self.train_epoch
        train_data=[]
        for i in train_epoch:
            if train_data==[]:
                train_data=i 
            else:
                train_data=np.concatenate((train_data,i),axis=1)
        maps, segmentation = mne_microstates.segment(train_data, n_states=num_states)
        length=len(train_epoch[0][0])
        seg=[]
        for i in range(len(train_epoch)):
            seg.append(segmentation[i*length:(i+1)*length])
        seg=np.array(seg)
        self.train_Feature.update(MicroState=seg)
        self.map=maps

    def get_micro_state(self):
        '''
        Args:
            None
        return:
            self.test_Feature["MicroState"]: test_data after transformation
        '''
        test_epoch=self.test_epoch
        map=self.map
        test_micro_state=[]
        test_data=[]
        for i in test_epoch:
            if test_data==[]:
                test_data=i 
            else:
                test_data=np.concatenate((test_data,i),axis=1)

        for i in range(len(test_data[0])):
            data_point=test_data[:,i]
            tmp=[]
            for j in range(len(map)):
                dis=np.linalg.norm(data_point - map[j])
                tmp.append(abs(dis))
            group=tmp.index(min(tmp))
            test_micro_state.append(group)
        length=len(test_epoch[0][0])
        seg=[]
        for i in range(len(test_epoch)):
            seg.append(test_micro_state[i*length:(i+1)*length])
        seg=np.array(seg)
        self.test_Feature.update(MicroState=seg)


    def CSP(self,train=True,feature=""):
        '''
        Args:
            train: whether it deals with training dataset or testing dataset
            feature: which feature we use to build the model
        return:
            self.model["CSP"]:
            self.train_Feature[feature]:
            self.
        '''

        if train==True:
            if len(feature)==0:
                feature="Original"
                data=self.train_FeaChannel["Original"]
                lab=self.train_lab

            elif len(feature)!=0:
                if (feature in self.train_FeaChannel)==False:
                    raise ValueError("this feature is not in the dataset")
                data=self.train_FeaChannel[feature]
                lab=self.train_lab

            csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
            csp_eeg_ori=csp.fit_transform(data, lab)
            scaler = StandardScaler()
            csp_eeg = scaler.fit_transform(csp_eeg_ori)
            clf = Pipeline([('CSP', csp),('standardize', scaler)])
            self.model.update(CSP=clf)
            self.train_Feature[feature]=csp_eeg

        if train==False:
            if len(feature)==0:
                feature="Original"
                data=self.test_FeaChannel["Original"]
                lab=self.test_lab

            elif len(feature)!=0:
                if (feature in self.test_FeaChannel)==False:
                    raise ValueError("this feature is not in the dataset")

                data=self.test_FeaChannel[feature]
                lab=self.test_lab

            model=self.model["CSP"]
            eeg_data=model.transform(data)
            self.test_Feature[feature]=eeg_data

    def EMS(self,feature="",train=True):#
        '''
        Transformer to compute event-matched spatial filters.
        This version of EMS operates on the entire time course. No time window needs to be specified. 
        The result is a spatial filter at each time point and a corresponding time course. 
        Intuitively, the result gives the similarity between the filter at each time point and the data vector (sensors) at that time point.
        Args:
        feature: which feature data used to constract EMS
        train: the data is training set and testing set
        returns:
        self.model: the ems model used to compute
        self.train_Feature:data after transformation
        '''
        if train==True:
            if len(feature)==0:
                feature="Original"
                data=self.train_FeaChannel["Original"]
                lab=self.train_lab

            elif len(feature)!=0:
                if (feature in self.train_FeaChannel)==False:
                    raise ValueError("this feature is not in the dataset")

                data=self.train_FeaChannel[feature]
                lab=self.train_lab

            ems = EMS()
            ems_eeg_ori=ems.fit_transform(data, lab)
            scaler = StandardScaler()
            ems_eeg = scaler.fit_transform(ems_eeg_ori)
            clf = Pipeline([('EMS', ems),('standardize', scaler)])
            self.model.update(EMS=clf)
            self.train_Feature[feature]=ems_eeg

        if train==False:
            if len(feature)==0:
                feature="Original"
                data=self.test_FeaChannel["Original"]
                lab=self.test_lab

            elif len(feature)!=0:
                if (feature in self.test_FeaChannel)==False:
                    raise ValueError("this feature is not in the dataset")

                data=self.test_FeaChannel[feature]
                lab=self.test_lab

            model=self.model["EMS"]
            ems_data=model.transform(data)
            self.test_Feature[feature]=ems_data


#######connectivity remain don't knw how to use

########################################classification model####################################
    def permutation(self,feature=""):
        #need to use the epoch_data (right after create epochs)
        '''
        permute the train data to avoid the 5 left,5 right pattern
        Args:
            feature:feature used in this model
        return:
            self.train_Feature: the data after permutation
            self.train_lab: the data label after permutation
        '''
        if len(feature)==0:
            feature="Original"

        if (feature in self.train_Feature)==False:
            raise ValueError("this feature is not in the dataset")

        data=self.train_Feature[feature]
        lab=self.train_lab
        per = np.random.permutation(data.shape[0])

        if len(data.shape)==3:#epoch data with data for each channel
            new_train_data=data[per,:,:]
            new_train_lab=lab[per]
            self.train_Feature[feature]=new_train_data
            self.train_lab=new_train_lab

        if len(data.shape)==2:#epoch data with data only for combined channel
            new_train_data=data[per,:]
            new_train_lab=lab[per]
            self.train_Feature[feature]=new_train_data
            self.train_lab=new_train_lab
    
    
    def PCA(self,feature=0,num_components=5,train=True):
        '''
        Do the PCA to reduce the dimension
        Para:
        data: except for using the original data to do the PCA, num_event*num_feature
        need to predefine which data/feature to do the PCA
        num_components: number of components/can be None: pertain all the none zero components
        Return:
        The lower dimension of the event with feature num_events*num_components
        '''  
        if train==True:
            if feature==0:
                feature="Original"
                data=self.train_Feature["Original"]
            else:
                if (feature in self.train_Feature)==False:
                    raise ValueError("this feature is not in the dataset")

                data=self.train_Feature[feature]
                pca=KernelPCA(n_components=num_components)
                data_transformed=pca.fit_transform(data)
                pca_model=pca.fit(data)
                self.train_Feature[feature]=data_transformed
                self.model.update(PCA=pca_model)

        elif train==False:
            if feature==0:
                data=self.test_Feature["Original"]
            else:
                if (feature in self.test_Feature)==False:
                    raise ValueError("this feature is not in the dataset")
                data=self.test_Feature[feature]
                model=self.model["PCA"]
                data_transformed=model.fit_transform(data)
                self.test_Feature[feature]=data_transformed


    
    def LDA(self,feature=''):
        '''
        train the LDA model
        Args:
        feature: the data under which feature to train the data
        return: 
        self.model: add LDA model into the model set
        '''
        if len(feature) == 0:#we only use one feature(the original data or micro state)
            feature="Original"
        else:
            feature=feature
        if (feature in self.train_Feature)==False:
            raise ValueError("this feature is not in the dataset")

        data=self.train_Feature[feature]
        lab=self.train_lab

        clf=LinearDiscriminantAnalysis()   
        clf.fit(data,lab)
        self.model.update(LDA=clf)
        #scores=cross_val_score(clf,data,lab,cv=5)
        #print(scores)


    def SVM(self,feature=""):
        '''
        train the SVM model
        Args:
        feature: the data under which feature to train the data
        return: 
        self.model: add SVM model into the model set
        '''

        if len(feature) == 0:#we only use one feature(the original data or micro state)
            feature="Original"
        else:
            feature=feature
        
        if (feature in self.train_Feature)==False:
            raise ValueError("this feature is not in the dataset")

        data=self.train_Feature[feature]
        lab=self.train_lab

        model = SVC(kernel='rbf', probability=True)    
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}    
        grid_search = GridSearchCV(model, param_grid, n_jobs = 8, verbose=1)    
        grid_search.fit(data,lab)    
        best_parameters = grid_search.best_estimator_.get_params()    
        svm = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)    
        svm.fit(data,lab)
        self.model.update(SVM=svm)
        #scores=cross_val_score(svm,data,lab,cv=5)
        #print(scores)

    def RF(self,feature=""):
        '''
        train the Random Forest model
        Args:
        feature: the data under which feature to train the data
        return: 
        self.model: add Random Forest model into the model set
        '''

        if len(feature)==0:
            feature="Original"
        else:
            feature=feature

        if (feature in self.train_Feature)==False:
            raise ValueError("this feature is not in the dataset")

        data=self.train_Feature[feature]
        lab=self.train_lab

        model = RandomForestClassifier(n_estimators=10,oob_score=True)   
        param_grid = {'n_estimators': [10, 100, 300], 'criterion': ["gini", "entropy","log_loss"],"max_depth":[10,100,300],"max_leaf_nodes":[10,20,50]}    
        grid_search = GridSearchCV(model, param_grid, n_jobs = -1, verbose=1)    
        grid_search.fit(data,lab)    
        best_parameters = grid_search.best_estimator_.get_params()    
        model = RandomForestClassifier(n_estimators=best_parameters["n_estimators"], criterion=best_parameters['criterion'], max_depth=best_parameters['max_depth'],max_leaf_nodes=best_parameters["max_leaf_nodes"])    
        model.fit(data,lab)
        #print(best_parameters)
        self.model.update(RF=model)
        #scores=cross_val_score(model,data,lab,cv=5)
        #print(scores)

    def GB(self,feature=""):# GradientBoosting
        '''
        train the GradientBoosting model
        Args:
        feature: the data under which feature to train the data
        return: 
        self.model: add GradientBoosting model into the model set
        '''

        if len(feature)==0:
            feature="Original"
        else:
            feature=feature
        
        if (feature in self.train_Feature)==False:
            raise ValueError("this feature is not in the dataset")

        data=self.train_Feature[feature]
        lab=self.train_lab

        model=GradientBoostingClassifier()
        param_grid = {'n_estimators': [10, 100], 'learning_rate': [0.1,0.2],"max_depth":[10,100,300],"max_leaf_nodes":[10,20,50]}  
        grid_search = GridSearchCV(model, param_grid, n_jobs = -1, verbose=1)    
        grid_search.fit(data,lab)    
        best_parameters = grid_search.best_estimator_.get_params()    
        model = GradientBoostingClassifier(n_estimators=best_parameters["n_estimators"], criterion=best_parameters['criterion'], max_depth=best_parameters['max_depth'],max_leaf_nodes=best_parameters["max_leaf_nodes"])    
        model.fit(data,lab)
        #print(best_parameters)
        self.model.update(GB=model)
        #scores=cross_val_score(model,data,lab,cv=5)
        #print(scores)


    
    def predict(self,model="",feature='',add_model=None):
        '''
        use differnet model to predict
        Args:
            model: which model used to predict
            feature: which feature used to predict
            add_model:whether to use another model to do the prediction
        '''
        if len(model)==0:
            raise DimentionalError("need to put the type of model")

        if (model in self.model)==False:
            raise ValueError("this feature is not in the dataset")
    
        if len(feature) == 0:#we only use one feature(the original data or micro state)
            feature="Original"
        else:
            feature=feature

        if (feature in self.test_Feature)==False:
            raise ValueError("this feature is not in the dataset")

        data=self.test_Feature[feature]
        lab=self.test_lab
        if add_model==None:
            model=self.model[model]
        else:
            model=add_model
        res=model.score(data,lab)
        pre=model.predict(data)
        acc=0
        for i in range(len(pre)):
            if pre[i]==lab[i]:
                acc=acc+1
        acc=acc/len(pre)

        return [res,acc]
