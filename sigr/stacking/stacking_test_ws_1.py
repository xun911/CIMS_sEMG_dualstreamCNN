import os
import scipy.io as sio
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode as Mode


#prob_root = 'Z:/wwt/stacking-feature'
prob_root = '/home/weiwentao/public-2/wwt/stacking-feature'
window=40
stride=1
dataset_name = 'capgmyo-multistream'
mode='average'



def get_probs_labels(fold, window, stride, mode):
    data_path = (('%s/TEST-%s-softmax-fold-%d') % (prob_root, dataset_name, fold))
    
    train_prob_path = os.path.join(data_path, 'train_prob.mat')
    train_label_path = os.path.join(data_path, 'train_true.mat')
    train_segment_path = os.path.join(data_path, 'train_segment.mat')
    
    test_prob_path = os.path.join(data_path, 'test_prob.mat')
    test_label_path = os.path.join(data_path, 'test_true.mat')
    test_segment_path = os.path.join(data_path, 'test_segment.mat')
    
    train_prob = sio.loadmat(train_prob_path)['data'].astype(np.float32)
    train_label = sio.loadmat(train_label_path)['data'].astype(np.float32)
    train_segment = sio.loadmat(train_segment_path)['data'].astype(np.float32)
    
    test_prob = sio.loadmat(test_prob_path)['data'].astype(np.float32)
    test_label = sio.loadmat(test_label_path)['data'].astype(np.float32)
    test_segment = sio.loadmat(test_segment_path)['data'].astype(np.float32)
    
    train_label = np.transpose(train_label)
    train_segment = np.transpose(train_segment)

    test_label = np.transpose(test_label)
    test_segment = np.transpose(test_segment)    
    
    assert (len(train_label)==len(train_segment)==train_prob.shape[0])    
    assert (len(test_label)==len(test_segment)==test_prob.shape[0])    
    
    train_probs, train_labels=get_trial_probs(train_prob, train_label, train_segment, window, stride, mode)
    test_probs, test_labels=get_trial_probs(test_prob, test_label, test_segment, window, stride, mode)
    
    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)
    
    train_labels = np.reshape(train_labels, (train_labels.shape[0],))
    test_labels = np.reshape(test_labels, (test_labels.shape[0],))
    
    return train_probs, train_labels, test_probs, test_labels
    
def get_trial_probs(prob, label, segment, window, stride, mode):
    
    segment_idx = np.unique(segment)
    
    probs = []
    labels = [] 
    segment = np.reshape(segment, (segment.shape[0],))
    for i in range(len(segment_idx)):
        trial_prob = prob[segment==segment_idx[i]]
        trial_label = label[segment==segment_idx[i]]
        trial_prob, trial_label = get_trial_segments(trial_prob, trial_label, window, stride, mode=mode)
        probs.append(trial_prob)
        labels.append(trial_label)
        
    probs = np.vstack(probs)
    labels = np.vstack(labels)

    return probs, labels    
           

def get_trial_segments(trial_prob, trial_label, window, stride, mode=None):
    
    assert (len(np.unique(trial_label)) == 1)
    assert (len(trial_label)==trial_prob.shape[0])    
    
    ch_num = trial_prob.shape[1] 
    assert (trial_prob.shape[0] >= window)
    trial_prob = get_segments(trial_prob, window, stride)
    sample_num = trial_prob.shape[0]             
    
    trial_label = np.ones((sample_num,1))*np.unique(trial_label) 
    
    if mode == 'average':
         result = []
         trial_prob = trial_prob.reshape(-1, window, ch_num)
         for kk in trial_prob:
             result.append(np.mean(kk, axis=0))
         result = np.array(result)
         trial_prob = result
         del result
    elif mode == 'for_voting':     
         trial_prob = trial_prob.reshape(-1, window, ch_num)
    elif mode == 'for_voting_cut':   
         median_range = int(float(len(trial_prob)) / 4)                
         start_point = int(float(len(trial_prob)) / 2) - median_range
         end_point = start_point + median_range*2
         trial_prob = trial_prob[start_point:end_point]                                   
         trial_prob = trial_prob.reshape(-1, window, ch_num)          
         trial_label = trial_label[start_point:end_point]           
         
    return trial_prob, trial_label
        

def get_segments(data, window, stride):
    return windowed_view(
        data.flat,
        window * data.shape[1],
        (window-stride)* data.shape[1]
    )
        
def windowed_view(arr, window, overlap):
    from numpy.lib.stride_tricks import as_strided
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                  window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                   arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)        
        
          
def classify_single_fold(fold, window, stride, mode, classifier_type, normalization=False):
    
    if classifier_type == 'voting':
        mode = 'for_voting'
    elif  classifier_type == 'voting_cut':
        mode = 'for_voting_cut'
    train_probs, train_labels, test_probs, test_labels = get_probs_labels(fold, window, stride, mode)
   
    if classifier_type == 'linearSVM':
        if normalization:
            scaler  = StandardScaler().fit(train_probs) 
            train_probs = scaler.transform(train_probs)
            test_probs = scaler.transform(test_probs)        
        clf = svm.LinearSVC()
        clf.fit(train_probs, train_labels)
        test_predict = clf.predict(test_probs)
    elif classifier_type == 'rbfSVM':
        if normalization:
            scaler  = StandardScaler().fit(train_probs) 
            train_probs = scaler.transform(train_probs)
            test_probs = scaler.transform(test_probs)        
        clf = svm.SVC()
        clf.fit(train_probs, train_labels)
        test_predict = clf.predict(test_probs)
    elif classifier_type == 'random_forest':
        
        if normalization:
            scaler  = StandardScaler().fit(train_probs) 
            train_probs = scaler.transform(train_probs)
            test_probs = scaler.transform(test_probs)        
        
        clf = RandomForestClassifier(n_estimators=200)
        clf.fit(train_probs, train_labels)
        test_predict = clf.predict(test_probs)       
        
    elif classifier_type == 'voting':        
        test_predict = []          
        for prob in test_probs:
            predict_labels = np.argmax(prob,axis=1)
            voting_result = Mode(predict_labels)[0][0]
            test_predict.append(voting_result)
        test_predict = np.array(test_predict) 

    elif classifier_type == 'voting_cut':        
        test_predict = []          
        for prob in test_probs:
            predict_labels = np.argmax(prob,axis=1)
            voting_result = Mode(predict_labels)[0][0]
            test_predict.append(voting_result)
        test_predict = np.array(test_predict)
        
    
         
    
    num_true_predicts = float(np.sum(test_predict==test_labels)) 
    num_true_labels = float(len(test_labels))    
    
#    test_score = clf.score(test_probs, test_labels)    
    
    return num_true_predicts, num_true_labels

def classify(folds, window, stride, mode, classifier_type, normalization=False):
    total_preds = 0; 
    total_trues = 0
    for fold in folds:
        num_true_predicts, num_true_labels = classify_single_fold(fold=fold, 
                                                                      window=window, 
                                                                      stride=stride, 
                                                                      mode=mode, 
                                                                      classifier_type=classifier_type,
                                                                      normalization=normalization)
        total_preds = total_preds + num_true_predicts
        total_trues = total_trues + num_true_labels
        fold_score = float(num_true_predicts) / float(num_true_labels) * 100
        print ("Fold %d accurcay = %f " % (fold, fold_score))
        
    acc = float(total_preds) / float(total_trues)
    return acc                                                          

#a1,a2 = classify_single_fold(1, window=window, stride=stride, mode=None, classifier_type='voting', normalization=True)
#acc = a1/a2
#print acc
folds = range(18)
acc = classify(folds,window=window,stride=stride,mode=None, classifier_type='random_forest',normalization=True)
print acc