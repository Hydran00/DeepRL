import gym
from gym import spaces
# import pygame
import numpy as np
import torchvision
import torch
from functools import total_ordering
import clip
import pandas as pd
import os
import gdown
import tarfile
import json
from enum import Enum
from torchvision import transforms
from torch.utils.data import Dataset
import gdown
from PIL import Image #, ImageDraw
import random
import torch.nn.functional as F
from collections import deque
import cv2
import numpy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HISTORY_LENGTH = 50
ACT_NUM = 9
INPUT_SIZE = 1024+1024+5+450
RNN_SIZE= 1024
IMG_TXT_EMB_SIZE= 1024
BBOX_EMB=1024
NUM_SEGMENTS=9
import math
@total_ordering
class Actions(Enum):
  ACT_RT = 0 #Right
  ACT_LT = 1 #Left
  ACT_UP = 2 #Up
  ACT_DN = 3 #Down
  ACT_TA = 4 #Taller
  ACT_FA = 5 #Fatter
  ACT_SR = 6 #Shorter
  ACT_TH = 7 #Thiner
  ACT_TR = 8 #Trigger

  def __lt__(self, other):
    if self.__class__ is other.__class__:
      return self.value < other.value

class VisualGroundingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    CONVERGENCE_THRESHOLD = 0.7
    def __init__(self,dataset, num_agent,agent_offset=0,render_mode=None):
        # self.window_size = 512  # The size of the PyGame window
        self.dataset = dataset
        self.idx = {"train": 0, "val": 0}
        self.iou = 0
        self.current_iou=0
        self.highest_iou=0
        self.avg_similarity=0
        self.num_agent=num_agent
        self.agent_offset=agent_offset
        self.idx["train"]= self.agent_offset
        self.idx["val"]= self.agent_offset
        # self._max_episode_steps=50
        self.steps_num=0
        # _, _, self.width, self.height = RefCOCOg[self.idx]
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(low=-100000, high=100000, shape=(INPUT_SIZE,))
        self.action_history = deque(maxlen = HISTORY_LENGTH)

        # We have 9 actions, corresponding to "right", "up", "left", "down", "v-shrink", "v-stretch", "h-shrink", "h-stretch", "confirm"
        self.action_space = spaces.Discrete(ACT_NUM)
        
        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        print("Env initialized")

    def _get_obs(self):
        return #{"agent": self._agent_location, "target": self._target_location}

    def _get_info(self, trigger_pressed):
        return {"pred_bbox": self._agent_location, "target_bbox": self._target_location, "trigger_pressed":trigger_pressed, "img_idx":self.idx[self.split]}

    def compute_vbbox(self):
      area_bbox = self.bbox_width * self.bbox_height
      area_img = self.width * self.height
      v_bbox = torch.tensor( [self.x1/(self.width-1), self.y1/(self.height-1), self.x2/(self.width-1), self.y2/(self.height-1), area_bbox/area_img])
      return v_bbox

    def history2vec(self):
      """
      Convert action history deque to vector for constructing state
      :param q: deque contains histories
      """
      history = np.array(self.action_history)
      res = np.zeros((HISTORY_LENGTH, ACT_NUM))
      if len(history) != 0:
        res[np.arange(len(history)) + (HISTORY_LENGTH - len(history)), history] = 1
      res = res.reshape((HISTORY_LENGTH * ACT_NUM)).astype(np.float32)
      return res
    
    def select_best_segment(self,image,sentences,n):
      width, height = image.size[0], image.size[1]
      sentences = clip.tokenize(sentences).to(DEVICE)
      similarities = []
      img_embeddings = []
      coordinates = []
      with torch.no_grad():
        sentences_emb = self.dataset.model.encode_text(sentences)
        # print(sentences_emb.shape)
        sentences_emb = torch.mean(sentences_emb, dim=0,dtype=float).unsqueeze(0)
        for i in range(n):
          for j in range(n):
            x1,y1,x2,y2 = j*width//n, i*height//n, (j+1)*width//n , (i+1)*height//n
            coordinates.append((x1,y1,x2,y2))
            image_emb = self.dataset.preprocess(image.crop((x1,y1,x2,y2))).unsqueeze(0).to(DEVICE)
            image_emb = self.dataset.model.encode_image(image_emb).to(DEVICE)
            img_embeddings.append(image_emb)
            # compute similarity between image and sentence
            similarities.append(torch.cosine_similarity(image_emb, sentences_emb,dim=-1).item())
        #retrieve the segment with the highest similarity
        # print("Similarities : ",similarities, "choosing segment ",np.argmax(similarities)," with similarity ",np.max(similarities))
        return img_embeddings[np.argmax(similarities)].squeeze(), coordinates[np.argmax(similarities)]
    
    def reset(self, seed=None, options=None):
        self.split = "train"
        if options is not None:
           self.split = options["split"]
        print("IOU : ",str(round(100*self.current_iou,3)),"% with ",self.steps_num) #| AVG_SIMILARITY: ",str(round(100*self.avg_similarity,3))+"%")
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        
        embeddings, bbox, width, height, image, sentences = self.dataset[self.idx[self.split],self.split]
        self.sentences = sentences
        self.image=image
        self.img_txt_emb = embeddings
        # dividing image into NUM_SEGMENTS segments (j*k lines) and choosing the one with the highest similarity
        self.bbox_emb, (self.x1,self.y1,self.x2,self.y2) = self.select_best_segment(self.image, self.sentences,int(math.sqrt(NUM_SEGMENTS)))
        # print("Image has shape ",self.image.size," and bbox_emb agent is in ", (self.x1,self.y1,self.x2,self.y2))
        self.width = width
        self.height = height
        # Choose the agent's location uniformly
        torch.set_printoptions(threshold=10_000)
        self.bbox_width = self.x2 - self.x1
        self.bbox_height = self.y2 - self.y1
        self._agent_location = torch.tensor([[self.x1, self.y1, self.x2, self.y2]]).to(DEVICE)

        v_bbox = self.compute_vbbox().to(DEVICE)
        bbox_x2 = bbox[0]+bbox[2]
        bbox_y2 = bbox[1]+bbox[3]
        self._target_location = torch.tensor([[bbox[0],bbox[1],bbox_x2,bbox_y2 ]]).to(DEVICE)
        self.current_iou = torchvision.ops.generalized_box_iou( self._agent_location ,self._target_location )[0].item()
        self.iou = self.current_iou
        self.highest_iou = self.current_iou
        self.action_history = deque(maxlen=HISTORY_LENGTH)
        
        v_hist = torch.zeros(ACT_NUM*HISTORY_LENGTH).to(DEVICE)
        # self.rnn_state = torch.zeros(RNN_SIZE * 2)
        state = torch.cat( (self.img_txt_emb.to(DEVICE),self.bbox_emb,v_bbox,v_hist) ).to(DEVICE)
        self.steps_num = 0
        # self.idx = self.idx + 1 
        info = self._get_info(False)
        self.idx[self.split] +=self.num_agent
        state =state.detach().cpu().numpy()
        return state.squeeze(), info

    def _update_bbox(self, action):
      ALPHA = 0.2
      BETA  = 0.1
      self.x2 = self.x1 + self.bbox_width
      self.y2 = self.y1 + self.bbox_height

      assert action >= Actions.ACT_RT.value and action <= Actions.ACT_TR.value
      if action <= Actions.ACT_DN.value:
        delta_w = int(ALPHA * self.bbox_width)
        delta_h = int(ALPHA * self.bbox_height)
      else:
        delta_w = int(BETA * self.bbox_width)
        delta_h = int(BETA * self.bbox_height)

      # assert delta_h != 0
      # assert delta_w != 0

      # PREVENT_STUCK:
      if (delta_h == 0):
        delta_h = 2
      if (delta_w == 0):
        delta_w = 2

      if action == Actions.ACT_RT.value:
        self.x1 += delta_w
        self.x2 += delta_w
      elif action == Actions.ACT_LT.value:
        self.x1 -= delta_w
        self.x2 -= delta_w
      elif action == Actions.ACT_UP.value:
        self.y1 -= delta_h
        self.y2 -= delta_h
      elif action == Actions.ACT_DN.value:
        self.y1 += delta_h
        self.y2 += delta_h
      elif action == Actions.ACT_TA.value:
        self.y1 -= delta_h
        self.y2 += delta_h
      elif action == Actions.ACT_FA.value:
        self.x1 -= delta_w
        self.x2 += delta_w
      elif action == Actions.ACT_SR.value:
        self.y1 += delta_h
        self.y2 -= delta_h
      elif action == Actions.ACT_TH.value:
        self.x1 += delta_w
        self.x2 -= delta_w
      elif action == Actions.ACT_TR.value:
        # print("Trigger after ",self.steps_num, " steps")
        pass
      else:
        raise ValueError('Invalid action')
      # print(f"{','.join([str(self.x1), str(self.y1), str(self.x2), str(self.y2)])}")

      # ensure bbox inside image
      if self.x1 < 0:
        self.x1 = 0
      if self.y1 < 0:
        self.y1 = 0
      if self.x2 < 0:
        self.x2 = 0
      if self.y2 < 0:
        self.y2 = 0
      if self.x1 >= self.width:
        self.x1 = self.width - 1
      if self.y1 >= self.height:
        self.y1 = self.height - 1
      if self.x2 >= self.width:
        self.x2 = self.width - 1
      if self.y2 >= self.height:
        self.y2 = self.height - 1
      # assert self.x1 < self.x2
      # assert self.y1 < self.y2

      # Ensure bbox doesn't get flipped
      x1 = min(self.x1, self.x2)
      y1 = min(self.y1, self.y2)
      x2 = max(self.x1, self.x2)
      y2 = max(self.y1, self.y2)
      if x1 ==x2:
          x2+=1
      if y1 ==y2:
          y2+=1

      self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

      return

    def step(self, action):
        self.steps_num+=1
        done = False
        reward = 0
        self.iou = self.current_iou

        self._update_bbox(action)
        # An episode is done iff the agent has reached the target

        # observation = self._get_obs()

        self.action_history.append(action)
        v_hist = torch.from_numpy(self.history2vec()).to(DEVICE)
        # print("action: ",action," -> current action hist: ",self.action_history)
        # print("Step: current vhist: ",v_hist)
        # if self.render_mode == "human":
        #     self._render_frame()
        self.bbox_width = self.x2 - self.x1
        self.bbox_height = self.y2 - self.y1
        self._agent_location =  torch.tensor([[self.x1, self.y1, self.x2, self.y2]]).to(DEVICE)
        v_bbox = self.compute_vbbox().to(DEVICE)
        
        agent_bbox_crop = self.image.crop( (self.x1, self.y1, self.x2, self.y2) )
        # self.img_txt_emb = RefCOCOg._get_vector(agent_bbox_crop, self.senteces)
        img = self.dataset.preprocess(agent_bbox_crop).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            self.bbox_emb = self.dataset.model.encode_image(img).squeeze(0).to(DEVICE)
        # state = torch.cat( (self.img_txt_emb.to(DEVICE),self.bbox_emb,v_bbox) ).to(DEVICE)
        state = torch.cat( (self.img_txt_emb.to(DEVICE),self.bbox_emb,v_bbox,v_hist) ).to(DEVICE)

        self.current_iou = torchvision.ops.generalized_box_iou( self._agent_location, self._target_location )[0].item()
        #Get reward
        # print("IOU computed",self.current_iou)
        # print("real = ",self._target_location)
        # print("agent = ",self._agent_location)

        # Creating embedding of current bbox state
        
        #compute average similarity between bbox and sentences
        # text_similarities = F.cosine_similarity(self.img_txt_emb, self.text_features)
        # bbox_similarity =  F.cosine_similarity(self.img_txt_emb, self.ground_truth_bbox_features)
        # similarities = torch.cat((text_similarities,bbox_similarity),dim=0)

        # self.avg_similarity = torch.mean(bbox_similarity).item()

        trigger_pressed = False
        # if self.current_iou > VisualGroundingEnv.CONVERGENCE_THRESHOLD or self.steps_num > 49 or action==Actions.ACT_TR.value:
        if self.steps_num >= 50 or action==Actions.ACT_TR.value:
            done = True
            if action==Actions.ACT_TR.value:
                trigger_pressed = True
            # if (self.current_iou > VisualGroundingEnv.CONVERGENCE_THRESHOLD):
            #   print(f"IOU% > {VisualGroundingEnv.CONVERGENCE_THRESHOLD*100}% ---> tot = {self.done_counter}")

        reward = torchvision.ops.generalized_box_iou( self._agent_location, self._target_location )[0].item()
        reward += self._calculate_reward(self.iou,self.current_iou,action==Actions.ACT_TR.value)

        reward += (-self.iou + 0.98 * self.current_iou)# + self.avg_similarity
        # print("agent in \n",self._agent_location, " target in :\n",self._target_location, "   with action ",action, " iou: ",self.current_iou)

        info = self._get_info(trigger_pressed)
        # state = state.to(DEVICE)
        state = state.detach().cpu().numpy()
        return state, reward, done, info


    def _calculate_reward(self, previous_iou, current_iou, is_stop_action):

      if is_stop_action:
        if current_iou > VisualGroundingEnv.CONVERGENCE_THRESHOLD:
          return 2.0  # Reward for correctly stopping when IoU has improved
        else:
          return -2.0  # Penalty for stopping when IoU has not improved
      else:
        iou_difference = current_iou - previous_iou
        if iou_difference > 0:
          return 0.1 + iou_difference  # Reward for an action that increases IoU
        else:
          return -0.1 + iou_difference  # Penalty for an action that decreases IoU

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
       return None

    def close(self):
        pass
        # if self.window is not None:
            # pygame.display.quit()
            # pygame.quit()
