from model import Agent
import numpy as np
import gym

env= gym.make('CartPole-v1')

agent=Agent(4,2)
agent.load()
avg_steps = []
evolutions = 1000
punishment = -1 #alter punishment to get better results

for i in range(evolutions):
    
    done=False
    step_lim = 0
    state = env.reset()
    state = np.reshape(state,[1,4])
    
    while not done:
        
        env.render()
        
        action = agent.act(state)
        
        next_state,reward,done,info = env.step(action)
        next_state = np.reshape(next_state,[1,4])
        
        if done:
            reward = punishment
        
        agent.memory(action,state,reward)
        
        step_lim+=1
        state=next_state
        
        if done:
            avg_steps.append(step_lim)
            if step_lim>=max(avg_steps):
                agent.save()
                print("\n\n\n\n {} \n\n\n\n".format(step_lim))
            print("EVOLUTION {} STEPS {}".format(i,step_lim))
            break
    agent.replay()
        
print("AVG_STEPS: {} MAX: {} MIN: {}".format(np.mean(avg_steps),max(avg_steps),min(avg_steps)))
env.close()
agent.save()
