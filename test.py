import gym
import torch
from utils import RefCOCOg
import argparse
from ppo_discrete_rnn import PPO_discrete_RNN
import torchvision
from normalization import Normalization, RewardScaling
import time as tm
import cv2
import numpy as np
from PIL import Image
VALIDATION_SET_SIZE=2572
def denormalize(img,x0_norm,y0_norm,x1_norm,y1_norm):
    width = img.shape[1]
    height = img.shape[0]
    # print("den",width,height,sep=" | ")
    x0 = int(x0_norm * width)
    y0 = int(y0_norm * height)
    x1 = int(x1_norm * width)
    y1 = int(y1_norm * height)
    return x0,y0,x1,y1

def compute_iou(agent, ground_truth):
    iou =  torchvision.ops.box_iou( agent, ground_truth)[0].item()
    print("iou : ",iou)
    return iou

action_dict = {
    0: 'right',
    1: 'left',
    2: 'up',
    3: 'down',
    4: 'taller',
    5: 'fatter',
    6: 'shorter',
    7: 'thinner',
    8: 'trigger'
}

class Test():
    def __init__(self, args, number):
        self.args = args
        gym.envs.register(
            id='VisualGrounding-v0',
            entry_point='env.env:VisualGroundingEnv'
            )
        self.number = number #id of model
        self.dataset = RefCOCOg("../","val")
        self.env = gym.make("VisualGrounding-v0",dataset=self.dataset,num_agent=1,random_validation=False)
        self.args.state_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.n
        self.args.episode_limit = self.env.max_steps_per_episode
        self.agent = PPO_discrete_RNN(self.args)
        self.agent.load_model("VisualGrounding-v0", self.number,self.args.steps_num)
        self.state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization


    def show_result(self,image,sentences,info):
        gt_x0,gt_y0,gt_x1,gt_y1 = int(info["target_bbox"][0][0].item()),int(info["target_bbox"][0][1].item()),int(info["target_bbox"][0][2].item()),int(info["target_bbox"][0][3].item())
        x0,y0,x1,y1 = int(info["pred_bbox"][0][0].item()),int(info["pred_bbox"][0][1].item()),int(info["pred_bbox"][0][2].item()),int(info["pred_bbox"][0][3].item())
        #draw predicted bbox
        bbox_img = cv2.rectangle(np.array(image), (int(x0), int(y0)), (int(x1), int(y1)), (255,0,0), 2)
        #draw ground truth
        bbox_img = cv2.rectangle(bbox_img,(int(gt_x0),int(gt_y0)),(int(gt_x1),int(gt_y1)), (0,0,255), 2)
        
        # bbox_img = cv2.resize(bbox_img, (int(bbox_img.shape[1]/3), int(bbox_img.shape[0]/3)))

        Image.fromarray(bbox_img).show()
        print("Groud Truth: {:>5} {:>5} {:>5} {:>5}  |    agent_location: {:>5} {:>5} {:>5} {:>5}".format(gt_x0,gt_y0,gt_x1,gt_y1,x0,y0,x1,y1))
        print("sentences:\n",sentences)
        tm.sleep(7)

    def evaluate(self):
        evaluate_reward = 0
        evaluate_iou, iou_count = 0, 0
        accuracy_count = 0
        for i in range(0,VALIDATION_SET_SIZE):
            episode_reward, done, info = 0, False, {}
            s, info = self.env.reset(options={"split":"val"})
            img_idx = info["img_idx"]
            _,_,_,_, image, sentences= self.dataset[img_idx,"val"]
            self.agent.reset_rnn_hidden()
            counter = 0
            actions=[]
            
            while info["trigger_pressed"]==False and  counter < 12:
                if self.args.use_state_norm:
                    s = self.state_norm(s)
                a, a_logprob = self.agent.choose_action(s, evaluate=True)
                s_, r, done, info = self.env.step(a)
                print("received reward: ",r)
                episode_reward += r
                s = s_
                counter+=1
                # print("Episode reward: ",episode_reward)
                actions.append(action_dict[a])
                if a == 8:
                    print("Trigger pressed after {} steps".format(counter))
                    # self.show_result(image,sentences,info)
            print("Episode reward: ",episode_reward)
            iou = torchvision.ops.box_iou(info["target_bbox"],info["pred_bbox"]).item()
            iou_count += 1
            if iou > 0.5:
                accuracy_count += 1
            evaluate_iou += iou
            print("#{: >5} ->BBOX_IOU: {: >10} | mean IOU: {: >10} | accuracy: {: >10}".format(i,round(iou, 4),round(evaluate_iou/iou_count,4),round(accuracy_count/iou_count,4)))
            print("Actions: ",actions)
            print("\n\n#######################################################################")
            evaluate_reward += episode_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("-n", type=int, default=int(-1), help=" name index")
    parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_model_freq", type=int, default=2e4, help="Save frequency")
    parser.add_argument("--evaluate_times", type=float, default=100, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.20, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_gru", type=bool, default=False, help="Whether to use GRU")
    parser.add_argument("--resume", type=bool, default=False, help="load last weights and resume training from checkpoint")
    parser.add_argument("--steps_num", type=int, default=-1, help="steps number to load weights")
    parser.add_argument("--transformer", type=bool, default=False, help="whether to use transformer instead of lstm")
    args = parser.parse_args()
    if args.n == -1:
        print("Please input the index of the model")
        exit()
    test = Test(args,args.n)
    test.evaluate()
