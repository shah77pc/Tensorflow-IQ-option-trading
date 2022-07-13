import numpy as np
import pandas as pd
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from iq import get_data_needed ,login ,get_data_neededOTC
from keras import backend as K 
import time
import os
from tapy import Indicators
try:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except Exception as e:
  # Memory growth must be set before GPUs have been initialized
  print(e)

SEQ_LEN = 8 # how long
FUTURE_PERIOD_PREDICT = 1 # how far into the future are we trying to predict

def classify(current,future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop("future", 1) 
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    indexes = df.index
    df_scaled = scaler.fit_transform(df)
    
    df = pd.DataFrame(df_scaled,index = indexes)
    
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences
            sequential_data.append([np.array(prev_days), i[-1]]) 

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if  put
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # if call
            buys.append([seq, target]) 

    random.shuffle(buys)  
    random.shuffle(sells)  # shuffle 

    
    lower = min(len(buys), len(sells))  

    buys = buys[:lower]  
    sells = sells[:lower]  
    
    
    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle

    X = []
    y = []

    for seq, target in sequential_data:  
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets

    return np.array(X), y  



def train_data():
    iq = login()

    # if os.path.exists(R"C:\Users\Advice\Downloads\binary-bot-master\savemodel\asd.h5"):
    #   os.remove(R"C:\Users\Advice\Downloads\binary-bot-master\savemodel\asd.h5")
    #actives = ['EURUSD','GBPUSD','EURJPY','AUDUSD']

    df = get_data_needed(iq)
    
    df.isnull().sum().sum() # there are no nans
    df.fillna(method="ffill", inplace=True)
    df = df.loc[~df.index.duplicated(keep = 'first')]
    
    df['future'] = df["close"].shift(-FUTURE_PERIOD_PREDICT) # future prediction
    
    #df['MA_20'] = df['close'].rolling(window = 20).mean() #moving average 20
    #df['MA_5'] = df['close'].rolling(window = 5).mean() #moving average 50
    
    
    df['L14'] = df['min'].rolling(window=14).min()
    df['H14'] = df['max'].rolling(window=14).max()
    df['%K'] = 100*((df['close'] - df['L14']) / (df['H14'] - df['L14']) ) #stochastic oscilator
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    df['EMA_5'] = df['close'].ewm(span = 5, adjust = False).mean()
    df['EMA_20'] = df['close'].ewm(span = 20, adjust = False).mean() #exponential moving average
    #df['EMA_50'] = df['close'].ewm(span = 50, adjust = False).mean()
    
    rsi_period = 14 
    chg = df['close'].diff(1)
    gain = chg.mask(chg<0,0)
    df['gain'] = gain
    loss = chg.mask(chg>0,0)
    df['loss'] = loss
    avg_gain = gain.ewm(com = rsi_period - 1, min_periods = rsi_period).mean()
    avg_loss = loss.ewm(com = rsi_period - 1, min_periods = rsi_period).mean()
    df['avg_gain'] = avg_gain
    df['avg_loss'] = avg_loss
    rs = abs(avg_gain/avg_loss)
    df['rsi'] = 100-(100/(1+rs)) #rsi index
    
    df = df.drop(columns = {'open','min','max','avg_gain','avg_loss','L14','H14','gain','loss'}) #drop columns that are too correlated or are in somehow inside others
    
    df = df.dropna()
    dataset = df.fillna(method="ffill")
    dataset = dataset.dropna()
    
    dataset.sort_index(inplace = True)
    
    main_df = dataset
    
    main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
    main_df.dropna(inplace=True)
    
    main_df['target'] = list(map(classify, main_df['close'], main_df['future']))
    
    main_df.dropna(inplace=True)
    
    main_df['target'].value_counts()
    
    main_df.dropna(inplace=True)
    
    main_df = main_df.astype('float32')
    
    times = sorted(main_df.index.values)
    last_5pct = sorted(main_df.index.values)[-int(0.1*len(times))]
    
    validation_main_df = main_df[(main_df.index >= last_5pct)]
    main_df = main_df[(main_df.index < last_5pct)]
    
    train_x, train_y = preprocess_df(main_df)
    validation_x, validation_y = preprocess_df(validation_main_df)
    
    print(f"train data: {len(train_x)} validation: {len(validation_x)}")
    print(f"sells: {train_y.count(0)}, buys: {train_y.count(1)}")
    print(f"VALIDATION sells: {validation_y.count(0)}, buys : {validation_y.count(1)}")
    
    train_y = np.asarray(train_y)
    validation_y = np.asarray(validation_y)
    
    
    
    
    LEARNING_RATE = 0.001 #isso mesmo
    EPOCHS = 30  # how many passes through our data #20 was good
    BATCH_SIZE = 16  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
    NAME = f"{LEARNING_RATE}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-{EPOCHS}-{BATCH_SIZE}-PRED-{int(time.time())}"  # a unique name for the model
    print(NAME)
    
    
    try:
      gpus = tf.config.experimental.list_physical_devices('GPU')
      if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except Exception as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)
    
    earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.
    
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(2, activation='softmax'))
    
    
    opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE, decay=5e-15)
    
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    
    checkpoint = ModelCheckpoint(R"/kaggle/working/iqoptionapi/Tensorflow-IQ-option-trading/Tensorflow-IQ-option-trading/savemodel/asd.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') # saves only the best ones
    
    # Train model
    history = model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(validation_x, validation_y),
        callbacks=[checkpoint, earlyStoppingCallback],
    
    )
    return validation_x,validation_y
    K.clear_session()
import datetime 
i = 0
while True:
  if datetime.datetime.now().second == 0:
    os.system('cls')
    X = train_data()
    model = tf.keras.models.load_model(R"/kaggle/working/iqoptionapi/Tensorflow-IQ-option-trading/Tensorflow-IQ-option-trading/savemodel/asd.h5")
    score = model.evaluate(X[0],X[1], verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    model1 = tf.keras.models.load_model(R"/kaggle/working/iqoptionapi/Tensorflow-IQ-option-trading/Tensorflow-IQ-option-trading/savemodel/asd.h5")
    score1 = model1.evaluate(X[0],X[1], verbose=0)
    print(f'Test loss: {score1[0]} / Test accuracy: {score1[1]}')  
    if score[1] > 0.6:
      model.save(R'/kaggle/working/iqoptionapi/Tensorflow-IQ-option-trading/Tensorflow-IQ-option-trading/savemodel/asd.h5')
      model.save(R'/kaggle/working/iqoptionapi/Tensorflow-IQ-option-trading/Tensorflow-IQ-option-trading/savemodel/asd1.h5')
    elif score1[1] > 0.6:
      model1.save(R'/kaggle/working/iqoptionapi/Tensorflow-IQ-option-trading/Tensorflow-IQ-option-trading/savemodel/asd.h5')
      model1.save(R'/kaggle/working/iqoptionapi/Tensorflow-IQ-option-trading/Tensorflow-IQ-option-trading/savemodel/asd1.h5')
    elif score[1] < 0.6 and score1[1] < 0.6 and os.path.exists(R'/kaggle/working/iqoptionapi/Tensorflow-IQ-option-trading/Tensorflow-IQ-option-trading/savemodel/asd1.h5'):
      os.remove(R'/kaggle/working/iqoptionapi/Tensorflow-IQ-option-trading/Tensorflow-IQ-option-trading/savemodel/asd1.h5')
    i = i + 1
    print('round = ',i)


    
