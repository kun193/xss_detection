import time
from keras.models import Sequential
from keras.layers import Dense,InputLayer,Dropout,GRU
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import load_model
from processing import build_dataset
from utils import init_session
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from load_drebin import load_data
import pandas as pd

init_session()
batch_size=350
epochs_num=1
process_datas_dir="./file/process_data.pickle"
log_dir="./log/GRU.log"
model_dir="./file/GRU_model"

def test(model_dir,test_generator,test_size,input_num,dims_num,batch_size,training=True):
    print("Start Train Job! ")
    start = time.time()
    inputs = InputLayer(input_shape=(input_num, dims_num), batch_size=batch_size)
    layer1 = GRU(128)
    output = Dense(2, activation="softmax", name="Output")
    optimizer = Adam()
    model = Sequential()
    model.add(inputs)
    model.add(layer1)
    model.add(Dropout(0.5))
    # model.add(Dense(2,activation="softmax",name="Output"))
    model.add(output)
    if training==True:
        model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        # model.fit(train_generator, batch_size=32, epochs=40)
        model.fit_generator(train_generator, steps_per_epoch=train_size // batch_size, epochs=epochs_num)
        # model.fit_generator(train_generator, steps_per_epoch=5, epochs=5, callbacks=[call])
        # model.save(model_dir)
        model.save_weights('GRU_29.h5')
        end = time.time()
        print("Over train job in %f s" % (end - start))
        print(model.summary())
    else:
        model.load_weights('GRU_29.h5')

    labels_pre = []
    labels_true = []
    batch_num = test_size // batch_size + 1
    steps = 0
    for batch, labels in test_generator:
        if len(labels) == batch_size:
            labels_pre.extend(model.predict_on_batch(batch))
        else:
            batch = np.concatenate((batch, np.zeros((batch_size - len(labels), input_num, dims_num))))
            labels_pre.extend(model.predict_on_batch(batch)[0:len(labels)])
        labels_true.extend(labels)
        steps += 1
        print("%d/%d batch" % (steps, batch_num))
    labels_pre = np.array(labels_pre).round()

    def to_y(labels):
        y = []
        for i in range(len(labels)):
            if labels[i][0] == 1:
                y.append(0)
            else:
                y.append(1)
        return y

    y_true = to_y(labels_true)
    y_pre = to_y(labels_pre)
    acc = accuracy_score(y_true, y_pre)
    precision = precision_score(y_true, y_pre)
    recall = recall_score(y_true, y_pre)
    F1 = f1_score(y_true, y_pre)
    print("Accuracy score is :", acc)
    print("Precision score is :", precision)
    print("Recall score is :", recall)
    print("F1 score is :", F1)

if __name__=="__main__":
    train_generator, test_generator, train_size, test_size, input_num, dims_num=build_dataset(batch_size)
    #train(train_generator,train_size,input_num,dims_num)
    test(model_dir,test_generator,test_size,input_num,dims_num,batch_size)