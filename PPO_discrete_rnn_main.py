import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
import torchvision
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete_rnn import PPO_discrete_RNN
from env.env import *
from utils import *


class Runner:
    def __init__(self, args, env_name, number, seed, split="train"):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        self.dataset = None
        self.save_model_freq = 10000
        self.save_counter = 0
        self.just_saved=-1
        if split == "train":
            self.set_train()
        elif split == "val":
            self.set_eval()

        # Create env
        self.env = gym.make(env_name,dataset=self.dataset,num_agent=1)
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # self.env.seed(seed)
        # self.env.action_space.seed(seed)

        self.args.state_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.n
        self.args.episode_limit = 50#self.env._max_episode_steps  # Maximum number of steps per episode
        print("env={}".format(env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("episode_limit={}".format(args.episode_limit))

        self.replay_buffer = ReplayBuffer(args)
        self.agent = PPO_discrete_RNN(args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}'.format(env_name, number))#, seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0

        if self.args.use_state_norm:
            print("------use state normalization------")
            self.state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)
        print("RUNNER INITIALIZED")

    def set_train(self, ):
        self.dataset = RefCOCOg(data_dir="../",split="train")

    def set_eval(self, ):
        self.dataset = RefCOCOg(data_dir="../",split="val")

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1
            _, episode_steps = self.run_episode()  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def run_episode(self, ):
        episode_reward = 0
        s,info = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        self.agent.reset_rnn_hidden()
        for episode_step in range(self.args.episode_limit):
            if self.args.use_state_norm:
                s = self.state_norm(s)
            a, a_logprob = self.agent.choose_action(s, evaluate=False)
            v = self.agent.get_value(s)
            s_, r, done, info = self.env.step(a)
            episode_reward += r

            if done and episode_step + 1 != self.args.episode_limit:
                dw = True
            else:
                dw = False
            if self.args.use_reward_scaling:
                r = self.reward_scaling(r)
            # Store the transition
            self.replay_buffer.store_transition(episode_step, s, v, a, a_logprob, r, dw)
            s = s_
            if done==True or info["trigger_pressed"]==True:
                break

        # An episode is over, store v in the last step
        if self.args.use_state_norm:
            s = self.state_norm(s)
        v = self.agent.get_value(s)
        self.replay_buffer.store_last_value(episode_step + 1, v)
        # SAVE MODEL
        self.save_counter = self.total_steps
        # 11932 % 2000 = 1932 -> 0>=1800 
        
        if self.total_steps % self.save_model_freq >=  (self.save_model_freq-130) and self.just_saved<=0:
            print("SAVING MODEL")
            self.agent.save_model(self.env_name, self.number, self.total_steps)
            self.just_saved=130
        else:
            self.just_saved-=1
        return episode_reward, episode_step + 1
    
    def evaluate_policy(self, ):
        print("EVALUATION START")
        evaluate_reward = 0
        evaluate_iou = 0
        iou_step = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, done, info = 0, False, {}
            s, info = self.env.reset()
            self.agent.reset_rnn_hidden()
            while info['trigger_pressed'] == False and done == False:
                if self.args.use_state_norm:
                    s = self.state_norm(s, update=False)
                a, a_logprob = self.agent.choose_action(s, evaluate=True)
                s_, r, done, info = self.env.step(a)
                episode_reward += r
                s = s_
            evaluate_reward += episode_reward
            iou = torchvision.ops.box_iou(torch.tensor(info["target_bbox"]),torch.tensor(info["pred_bbox"])).item()
            evaluate_iou += iou
            iou_step += 1
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        # Save the rewards and models
        self.writer.add_scalar('evaluate_mean_iou:{}'.format(self.env_name), evaluate_iou/iou_step, global_step=self.total_steps)
        np.save('./data_train/PPO_env_{}_number_{}.npy'.format(self.env_name, self.number,self.total_steps), np.array(self.evaluate_rewards))
        print("EVALUATION END")


if __name__ == '__main__':
    torch.set_default_device("cuda:0")
    gym.envs.register(
        id='VisualGrounding-v0',
        entry_point='env.env:VisualGroundingEnv'
    )
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("-n", type=int, default=int(1), help=" name index")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--evaluate_times", type=float, default=50, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=16, help="Minibatch size")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_gru", type=bool, default=False, help="Whether to use GRU")

    args = parser.parse_args()

    env_names = ['VisualGrounding-v0']
    env_index = 0
    for seed in [0, 10, 100]:
        runner = Runner(args, env_name=env_names[env_index], number=7, seed=seed)
        runner.run()
    

