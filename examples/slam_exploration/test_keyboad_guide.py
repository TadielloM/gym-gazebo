import gym, ray
import gym_gazebo
import sys, signal

if __name__ == "__main__":
    env = gym.make('GazeboSlamExploration-v0')

    observation = env.reset()

    def handler(signum, frame):
        print('Sigint detected, closing environment: ', signum)
        env.close()
        sys.exit(1)

    signal.signal(signal.SIGINT, handler)

    i = 0
    cumulated_reward= 0
    while True:
        action = input("Select an action (from 0 to 25): ")
        if (action == 'w'): #forward
            action = 0
        elif(action == 'a'): #left
            action = 11
        elif(action == 's'): #back
            action = 17
        elif(action == 'd'): #right
            action = 15
        elif(action == 'q'): #up
            action = 9
        elif(action == 'e'): #down
            action = 13
        try:
            action = int(action)
        except ValueError:
            print("Action non valid")
            pass
        print(action)
        if (isinstance(action, int)):
            if(action == 100):
                break
            elif( action >=0 and action <=25):
                i+=1
                next_state, reward, done, info  = env.step(action)
                point_counter = 0
                for p in next_state:
                    if p[3]==0:
                        point_counter +=1
                print("Execution number: {} Reward: {} Points: {}".format(i,reward,point_counter))
                cumulated_reward += reward
            else:
                print("Action non valid")
        else:
            print("Action non valid")
            pass
    
    print("Cumulated reward: ", cumulated_reward)
    env.close()

    