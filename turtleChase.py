#!/usr/bin/env python3
import rospy
from turtle_step import TurtleMover
from turtlesim.srv import Spawn
from turtlesim.srv import TeleportAbsolute
from std_srvs.srv import Empty
import numpy as np
import torch as T
from DQN_utils import *

class ChaseEnvironment:
    def __init__(self):
        self.turtle_mover = TurtleMover()
    
    def Reset(self):
        try:
            reset_service = rospy.ServiceProxy('/reset', Empty)
            response = reset_service()
            self.SpawnTarget()
            self.distance = self.calculateDistance()
            initial_state = [self.turtle_mover.current_pose.x,
                             self.turtle_mover.current_pose.y,
                             self.turtle_mover.current_pose.theta,
                             self.xT,
                             self.yT]
            
            return np.array(initial_state)
        except rospy.ServiceException as e:
            rospy.logwarn(e)
        

    def SpawnTarget(self):
        try:
            spawn_service = rospy.ServiceProxy("/spawn", Spawn)
            self.xT = np.random.random()*11.0
            self.yT = np.random.random()*11.0
            theta = -np.pi + np.random.random()*2*np.pi
            response = spawn_service(self.xT, self.yT, theta, 'turtle_target')
            # rospy.loginfo(response)
        except rospy.ServiceException as e:
            rospy.logwarn(e)

    def TeleportTarget(self):
        try:
            teleport_service = rospy.ServiceProxy('/turtle_target/teleport_absolute', TeleportAbsolute)
            self.xT = np.random.random()*11.0
            self.yT = np.random.random()*11.0
            theta = -np.pi + np.random.random()*2*np.pi
            response = teleport_service(self.xT, self.yT, theta)
        except rospy.ServiceException as e:
            rospy.logwarn(e)
    
    def ClearTraces(self):
        try:
            clear_service = rospy.ServiceProxy('/clear', Empty)
            response = clear_service()
        except rospy.ServiceException as e:
            rospy.logwarn(e)
    
    def calculateDistance(self):
        distance = (self.turtle_mover.current_pose.x - self.xT)**2 + (self.turtle_mover.current_pose.y - self.yT)**2
        return np.sqrt(distance)

    def calculateReward(self):
        return np.exp(-self.distance/5.0) - 1
    
    def step(self, action):
        new_state = self.turtle_mover.move(action)
        new_state.extend([self.xT, self.yT])
        self.distance = self.calculateDistance()
        done = False
        if self.distance < 0.5:
            done = True
        reward = self.calculateReward()
        return np.array(new_state), reward, done
        

if __name__=='__main__':
    agent = Agent(0.99, 5, [256, 256], 4, 5e-4, 0.1, mem_size=6000, batch_size=128, replace_target_every=1000)
    env = ChaseEnvironment()
    
    scores = []
    
    training_episodes = 1200
    max_steps = 200
    
    for ep in range(training_episodes):
        state = env.Reset()

        current_score = 0
        done = False
        step = 0

        while step < max_steps:
            action = agent.choose_action_eps_greedy(state)
            new_state, reward, done = env.step(action)
            step += 1

            if done:
                reward += 100
            current_score += reward

            agent.memory.save_transition(state, action, reward, new_state, done)
            
            if done:
                env.TeleportTarget()
                env.ClearTraces()
                new_state, _, done = env.step(0)

            agent.learn()
            agent.update_target()

            state = new_state
            
        scores.append(current_score)
        print(f'Episode: {ep+1} / {training_episodes} | Average Score: {np.mean(scores):.3f}')
    
    scores = np.array(scores)
    np.save('turtleChaseScores.npy', scores)
    T.save(agent.DQN_eval.state_dict(), 'turtleChaseDQN.h5')