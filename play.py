import numpy as np
from model import Agent
import gym

env= gym.make('CartPole-v1')

done=False

step_lim = 0
state = env.reset()
state = np.reshape(state,[1,4])
agent = Agent(4,2)
agent.load()

while step_lim<2000:
        action = agent.act(state)
        
        env.render()
        
        next_state,reward,done,info = env.step(action)
        next_state = np.reshape(next_state,[1,4])
        
        step_lim+=1
        state=next_state
        if done:
            print("STEPS {}".format(step_lim))
            break
            
env.close() 
