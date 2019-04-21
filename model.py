import numpy as np
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Agent():
    
    def __init__(self,input_states,output_states):
        
        self.hidden_layers = 24 #12 #24 #number of hidden layers of neural network
        self.discount = 0.99 #9 #discount_rate change at your convience
        self.lr = 0.0001
        self.input_states = input_states
        self.output_states = output_states
        
        self.model = self.model_build()
        self.states,self.rewards,self.actions = [],[],[]
       
    def model_build(self):
            
        model = Sequential()
        model.add(Dense(units=self.hidden_layers,activation='relu',input_dim=self.input_states))
        
        model.add(Dense(units=self.hidden_layers,activation='relu'))
        model.add(Dense(units=self.hidden_layers,activation='relu'))
        #model.add(Dense(units=self.hidden_layers,activation='relu'))
        #model.add(Dense(units=self.hidden_layers,activation='relu'))
        
        model.add(Dense(units=self.output_states,activation='sigmoid'))
        model.summary()    
            
        model.compile(loss ='categorical_crossentropy',optimizer=Adam(lr=self.lr))
        
        return model
    
    def reset(self):
        self.states,self.rewards,self.actions = [],[],[]
    
    def act(self,state):     
        action = self.model.predict(state)
        return np.argmax(action[0])
    
    def save(self):
        self.model.save_weights('model.h5')
        
    def load(self):
        return self.model.load_weights('model.h5')
    
    def memory(self,action,state,reward):
        self.actions.append(action)
        self.rewards.append(reward)
        self.states.append(state)
        
    def discount_rewards(self):
        
        dr = np.zeros_like(self.rewards)
        rdr = 0
        for t in reversed(range(len(self.rewards))):
            rdr = rdr * self.discount+self.rewards[t]
            dr[t] = rdr
        return dr
    def replay(self):
        
        evol_len = len(self.states)
        
        discount_reward = self.discount_rewards()
        discount_reward -= np.mean(discount_reward)
        discount_reward /= np.std(discount_reward)
        
        x = np.zeros((evol_len,self.input_states))
        y = np.zeros((evol_len,self.output_states))
        
        for i in range(evol_len):
            x[i] = self.states[i]
            y[i][self.actions[i]] = discount_reward[i]
            
        self.model.fit(x,y)    
        self.reset()
