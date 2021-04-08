import time
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
# from utils import GeneSeg
import csv,random,pickle
from load_drebin import load_data

batch_size=50
maxlen=200
vec_dir="file\\word2vec.pickle"
epochs_num=1
log_dir="log\\MLP.log"
model_dir="file\\SVM_model"
data_path = './data/feature_vectors'
label_path = './data/sha256_family.csv'

if __name__=="__main__":
    train_datas, train_labels, test_datas, test_labels = load_data(data_path,label_path,50000)
    print("Start Train Job! ")
    start = time.time()
    model = LinearSVC()
  #  model = SVC(C=1.0, kernel="linear")
    model.fit(train_datas,train_labels)
   # model.save(model_dir)
    end = time.time()
    print("Over train job in %f s" % (end - start))
    print("Start Test Job!")
    start=time.time()
    pre=model.predict(test_datas)
    end=time.time()
    print("Over test job in %s s"%(end-start))
    precision = precision_score(test_labels, pre)
    recall = recall_score(test_labels, pre)
    print("Precision score is :", precision)
    print("Recall score is :", recall)
    with open(model_dir,"wb") as f:
        pickle.dump(model,f,protocol=2)
    print("wirte to ",model_dir)