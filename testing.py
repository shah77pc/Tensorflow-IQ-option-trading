from threading import Timer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import datetime
import time
import os
from tensorflow.tools.docs.doc_controls import T
from iq import fast_data,higher,lower,login,checkwin,get_1candles,changepractice,changereal
from training import train_data
import tensorflow as tf
import sys

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

def preprocess_prediciton(iq):
    Actives = ['EURUSD','GBPUSD','EURJPY','AUDUSD']
    active = 'AUDUSD'
    main = pd.DataFrame()
    current = pd.DataFrame()
    for active in Actives:
        if active == 'AUDUSD':
            main = fast_data(iq,active).drop(columns = {'from','to'})
        else:
            current = fast_data(iq,active)
            current = current.drop(columns = {'from','to','open','min','max'})
            current.columns = [f'close_{active}',f'volume_{active}']
            main = main.join(current)
    
    df = main
    
    """
    graphical analysis components
    """
    
    df.isnull().sum().sum() # there are no nans
    df.fillna(method="ffill", inplace=True)
    df = df.loc[~df.index.duplicated(keep = 'first')]
    
    # df['EMA_5'] = df['close'].ewm(span = 5, adjust = False).mean()
    #df['MA_20'] = df['close'].rolling(window = 20).mean()
    #df['MA_5'] = df['close'].rolling(window = 5).mean()
    
    
    df['L14'] = df['min'].rolling(window=14).min()
    df['H14'] = df['max'].rolling(window=14).max()
    df['%K'] = 100*((df['close'] - df['L14']) / (df['H14'] - df['L14']) )
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    df['EMA_5'] = df['close'].ewm(span = 5, adjust = False).mean()
    df['EMA_20'] = df['close'].ewm(span = 20, adjust = False).mean()
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
    df['rsi'] = 100-(100/(1+rs))
    
    """
    Finishing preprocessing
    """
    df = df.drop(columns = {'open','min','max','avg_gain','avg_loss','L14','H14','gain','loss'})
    
    df = df.dropna()
    df = df.fillna(method="ffill")
    df = df.dropna()
    
    df.sort_index(inplace = True)
    scaler = MinMaxScaler()
    indexes = df.index
    df_scaled = scaler.fit_transform(df)
    
    pred = pd.DataFrame(df_scaled,index = indexes)

    sequential_data = []
    prev_days = deque(maxlen = SEQ_LEN)            
    
    for i in pred.iloc[len(pred) -SEQ_LEN :len(pred)   , :].values:
        prev_days.append([n for n in i[:]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days)])

    X = []

    for seq in sequential_data:
        X.append(seq)
    
    
    return np.array(X)
    
SEQ_LEN = 8
FUTURE_PERIOD_PREDICT = 1 

# train_data()
# model = tf.keras.models.load_model(R"C:\Users\Advice\Downloads\binary-bot-master\savemodel\asd.h5")
iq = login()
i = 1
startM = 120
bet_money = startM
bets = []
trade = False
ratio = 'AUDUSD'
martingale = 2
maxlose = 0
wait = 0
A = 0
reset = 0
ptc = 0
changepractice(iq)
while(1):
    
    # if i%2 == 0:
    #     train_data()
    #     model = tf.keras.models.load_model(R"C:\Users\Advice\Downloads\binary-bot-master\savemodel\asd.h5")
    #     i = i + 1 
    #     print('bet_money = ',bet_money)
    try: 
        if datetime.datetime.now().second == 0 and os.path.exists(R'C:\Users\Advice\Desktop\jkl\binary-bot-master\asd1.h5'): 
            time_taker = time.time()
            model = tf.keras.models.load_model(R'C:\Users\Advice\Desktop\jkl\binary-bot-master\asd1.h5')
            pred_ready = preprocess_prediciton(iq)             
            pred_ready = pred_ready.reshape(1,SEQ_LEN,pred_ready.shape[3])      
            result = model.predict(pred_ready)
            print('probability of PUT: ',result[0][0])
            print('probability of CALL: ',result[0][1])
            if trade:
                if checkwin(iq,id) == True:
                    # changereal(iq)
                    # if ptc == 0:
                    #     bet_money = startM
                    #     maxlose = 0
                    # ptc = 0
                    bet_money = startM
                    # maxlose = 0
                    print('WIN')
                elif checkwin(iq,id) == False:
                    # changepractice(iq)
                    # if ptc == 0:
                    #     bet_money = int(bet_money * 2)
                    #     maxlose = maxlose + 1
                    #     ptc = 1
                    # if maxlose == 5:
                    #     bet_money = startM
                    #     maxlose = 0
                    bet_money = int(bet_money / 2)
                    if bet_money < 30:
                        changepractice(iq) 
                    print('loose')
            data = get_1candles(iq,ratio)
            if result[0][0] > 0.6:
                while datetime.datetime.now().second < 30:
                    data = get_1candles(iq,ratio)
                    if data.iloc[-1]['Close'] > data.iloc[-2]['Close']:
                        id = lower(iq,bet_money,ratio)
                        print('PUT') 
                        trade = True
                        break
            elif result[0][1] > 0.6:
                while datetime.datetime.now().second < 30:
                    data = get_1candles(iq,ratio)
                    if data.iloc[-1]['Close'] < data.iloc[-2]['Close']:
                        id = higher(iq,bet_money,ratio) 
                        print('CALL')
                        trade = True
                        break
            else:
                trade = False 
            print(f'Time taken : {int(time.time()-time_taker)} seconds')
            if datetime.datetime.now().second >= 30:
                print('No trade')
                trade = False
        elif datetime.datetime.now().second == 0 and reset == 0:
            os.system('cls')
            print('confident < 0.6')
            reset = 1
        elif datetime.datetime.now().second == 2:
            reset = 0
    except:
        print('Error')
        time.sleep(10)

