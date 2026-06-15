import ast
import os
import pdb
import random
import copy
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import numpy as np
import pickle
import re
from enum import Enum
import time

from vico.agents.agent import Agent
from agents.memory import SemanticMemory, EventInstance
from agents.sentinel_challenge.base_nav import *
from agents.sg.builder.builder import Builder, BuilderConfig


class CoelaMeetingAgent(BaseNavigationMeetingAgent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=1, num_agents=1, enable_danger_zone=False):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger, lm_source, lm_id, max_tokens, temperature, top_p, init_generator, detect_interval, num_agents, enable_danger_zone)
        self.react_freq = 900 # 15min
        if self.debug:
            self.react_freq = 300 # 5 min for debug
        if self.no_react:
            self.react_freq = 1e8
        self.chat_time_limit = 10 # 10 seconds

        self.chatting_with: str = self.scratch["chatting_with"] if "chatting_with" in self.scratch else None # name
        self.chatting_buffer: list[list[datetime, list, str]] = self.scratch["chatting_buffer"] if "chatting_buffer" in self.scratch else []
        for chat in self.chatting_buffer:
            chat[0] = datetime.strptime(chat[0], "%B %d, %Y, %H:%M:%S")
        self.react_mode = None
        self.react_history = []
        self.last_react_time = self.curr_time
        self.last_go_time = self.curr_time
        self.task_complete = False
        self.goal_place = None
        self.sleep_time = 0
        self.banned = False

    def reset(self, name, pose):
        super().reset(name, pose)

    # def _process_obs(self, obs):

    def _process_obs(self, obs):
        super()._process_obs(obs)
        
        
        if self.chatting_with is not None:
            if self.chatting_with[0] == "someone":
                subject = self.s_mem.get_name_from_position(self.chatting_with[1])
                if subject is not None:
                    self.chatting_with[0] = subject
                    for chats in self.chatting_buffer:
                        if chats[1][0] == "someone":
                            chats[1] = self.chatting_with
                else:
                    self.logger.error(f"No subject found for the speech event at {self.chatting_with[1]}.")
                    # Image.fromarray(obs['rgb']).save(os.path.join(self.storage_path, 'episodic_memory', f'img_{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}.png'))

            for event in obs['events']:
                if event["type"] == "speech":
                    if event["position"][:2] == self.pose[:2]: # ignore self speech
                        continue
                    subject = self.s_mem.get_name_from_position(event["position"]) # need to deal with more than 2 people chatting
                    if subject is None:
                        self.logger.warning(f"No subject found for the speech event at {event['position']}.")
                    else:
                        self.logger.info(f"{self.name} hears {subject} at {event['position']} says: {event['content']}")
                        if self.chatting_with[0] == "someone":
                            self.chatting_with = [subject, event["position"]]
                            for chats in self.chatting_buffer:
                                if chats[1][0] == "someone":
                                    chats[1] = self.chatting_with
                        if subject == self.chatting_with[0] or self.chatting_with[1] == event["position"]:
                            self.chatting_buffer.append([self.curr_time, self.chatting_with, event["content"]])
        
        start = time.time()
        # react[also save episodic mem] every react_freq seconds or new objects appear
        if len(obs['events']) > 0:
            for event in obs['events']:
                if event["type"] == "speech":
                    self.conversation_history.append(Message(self.curr_time, event["subject"], event["content"]))
                    if event["position"][:2] == self.pose[:2]:
                        continue
                    subject = self.s_mem.get_name_from_position(event["position"])
                    event["content"] = f"I heard {subject if subject is not None else 'somebody outside of my view'} at {event['position']} says: {event['content']}"
                    kws = [subject, event['type']]
                elif event["type"] == "app message":
                    if event["subject"] != self.name:
                        continue
                    kws = [event["type"]]
                    self.logger.info(f"received app message: {event['content']}, my last action is {self.last_action}")
                    if self.last_action['type']=="query_map_tool":
                        if self.last_action['arg1']=="query_route":
                            if event['content'] is None:
                                time_to_arrival = timedelta(hours=23, minutes=59, seconds=59)
                            else:
                                time_to_arrival = timedelta(seconds=int(event['content'].calc_time(pose=self.get_outdoor_pose())))
                            if self.goal_place==self.last_action["arg2"]:
                                self.navigation_plan=event['content']
                                self.last_route=event["content"]
                                self.last_estimated_arrival_time = self.curr_time + time_to_arrival
                            self.app_message_history.append(Message(self.curr_time, event["subject"], f"The estimated time from current pose to {self.last_action['arg2']} is {time_to_arrival}s"))
                            self.update_known_eta(
                                {
                                    self.last_action['arg2']:
                                    {
                                        self.name: str(time_to_arrival)
                                    }
                                })
                        elif self.last_action["arg1"]=="query_place":
                            self.s_mem.update_with_new_knowledge(event["content"])
                        elif self.last_action["arg1"]=="query_nearby":
                            self.places_buffer.extend(event['content'])
                elif event["type"] == "sentinel signal":
                    kws = [event["type"]]
                    if event['content']['arg2'] != self.name: continue
                    if event['content']['arg1'] == 'ban':
                        self.logger.info("I'm being banned...")
                        self.banned = True
                else:
                    kws = [event["type"]]

                if obs['rgb'] is not None:
                    img_path = os.path.join(self.storage_path, 'episodic_memory', f'img_{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}.png')
                    Image.fromarray(obs['rgb']).save(img_path)
                else:
                    img_path = None

                self.logger.debug(f"reacting to new events: {event['content']}")
                self.add_event(event["type"], self.curr_time, event["position"], obs['current_place'], kws, img_path, event["content"], None)
            self.last_react_time = self.curr_time

        if not self.no_react and (self.last_react_time is None or (self.last_react_time != self.curr_time and (self.curr_time - self.last_react_time).total_seconds() > self.react_freq)):
            if obs['rgb'] is not None:

                # todo: get the keywords
                donot_add = False

                img_path = os.path.join(self.storage_path, 'episodic_memory',
                                        f'img_{self.curr_time.strftime("%B %d, %Y, %H:%M:%S")}.png')
                Image.fromarray(obs['rgb']).save(img_path)
                if "gt_seg_idxc_to_info" in obs:
                    desc = f"I see {', '.join([object.name for object in self.s_mem.object_builder.get_curr_objects()])}."
                    kws = [object.name for object in self.s_mem.object_builder.get_curr_objects()]
                    if kws == []:
                        donot_add = True
                else:
                    desc = self.generate_captioning(f"Describe what you see in one sentence. Start with 'I see'.", img=img_path)
                    kws = []
                if not donot_add:
                    self.logger.debug(f"reacting: {desc}")
                    self.add_event("observation", self.curr_time, self.pose[:3], obs['current_place'], [], img_path, desc, None)
                    self.last_react_time = self.curr_time

        self.logger.debug(f"Process obs 3: {start}, {time.time()}")
        # processing sentinels
        values, counts = np.unique(self.obs['segmentation'], return_counts=True)
        freq = dict(zip(values, counts))
        values_labels, counts_labels = np.unique(self.s_mem.current_labels, return_counts=True)
        freq_labels = dict(zip(values_labels, counts_labels))
        self.logger.info(f"segmentation is {freq}, labels are {freq_labels}")
    
    def add_event(self, event_type, event_time, event_position, event_place, event_keywords, event_img, event_description, event_text_ft, event_poignancy=None, event_expiration=None):
        event_id = str(len(self.curr_events))
        self.logger.debug(f"adding new event, {event_id}: {event_description}")
        this_experience = EventInstance(event_id, event_type, event_time, event_time, event_position, event_place, event_keywords, event_img, event_description, event_poignancy, event_expiration)
        self.curr_events.append(this_experience)

    def _act(self, obs):
        if self.banned:
            if self.pose[0]>-1000:
                return {"type": "teleport", "arg1": [-1500., -1500.]}
            return {"type": "task_complete"}
        self.logger.debug(f"Current mode is {self.mode}, while the trigger is {self.discussion_trigger}, mode_time_counter is {self.mode_time_counter}")
        if self.curr_time.second % 60 == 0 and self.curr_time.minute % 3 == 0:
            self.last_react_time = self.curr_time
        start = time.time()

        if self.sleep_time > 0:
            self.sleep_time -= 1
            if self.sleep_time == 0:
                return {'type': 'wake', 'arg1': None} # wake up
            return {'type': 'sleep', 'arg1': None}
        
        action = None if not self.task_complete else {'type': 'task_complete', 'arg1': None}
        
        if self.goal_place is not None and (self.last_go_time + timedelta(seconds=60) > self.curr_time or self.last_react_time!=self.curr_time):
            action, arrived = self.city_navigate(self.goal_place)
            if arrived:
                self.goal_place = None
                self.last_react_time = self.curr_time + timedelta(seconds=1)
            if action is not None:
                self.last_action = action
                return action
        
        utterance = None
        if self.chatting_with is not None:
            utterance = self.generate_utterance()
            action = self.conversation(self.chatting_with, utterance)
            self.logger.debug(f"Generate conversation action time: {time.time() - start}")
            if action is not None:
                return action
            action = {'type': 'wait', 'arg1': None}
            utterance = None # no conv
        elif not self.no_react and self.last_react_time == self.curr_time:
            utterance = self.generate_utterance()
            if utterance is None:
                self.logger.warning(f"Failed to generate utterance.")
                return {"type": "wait", "arg1": None}

        # react to the curr_events related retrieved events
        if not self.no_react and self.last_react_time == self.curr_time:
            self.react_mode, react_target = self.generate_react_mode(self.curr_events, utterance)
            self.goal_place = None
            self.task_complete = False
            self.logger.debug(f"The generated react is {self.react_mode} {react_target}")
            
            if self.react_mode == "speak":
                self.chatting_buffer = []
                self.chatting_with = None
                if utterance in ["null", 'None']:
                    self.logger.info(f"{self.name} stops the conversation.")
                    self.react_mode = "wait"
                    return {"type": "wait", "arg1": None}
                return self.conversation("someone", utterance)
            elif self.react_mode == "go":
                if react_target.startswith('<') and react_target.endswith('>'):
                    react_target = react_target[1:-1]
                self.goal_place = react_target
                self.meeting_place = react_target #need meeting place to track the destination in prompt.
                self.last_route = Route()
                self.last_nav = []
                action, arrived = self.city_navigate(self.goal_place)
                self.last_go_time = self.curr_time
                self.last_action = action
                return self.last_action
            elif self.react_mode == "query":
                if react_target.startswith('<') and react_target.endswith('>'):
                    react_target = react_target[1:-1]
                action = {"type": "query_map_tool", "arg1": "query_place", "arg2": react_target}
                self.last_action = action
                return self.last_action
            elif self.react_mode == "wait":
                self.last_action = {
                    'type': 'wait',
                    'arg1': None
                }
                return self.last_action
            elif self.react_mode == "complete task":
                self.task_complete = True
                self.last_action = {
                    'type': 'task_complete',
                    'arg1': None
                }
                return self.last_action
            else:
                self.logger.warning(f"Unknown react mode {self.react_mode}.")
                return None

        self.last_action=action
        return self.last_action

    def end_conversation(self):
        self.logger.info(f"{self.name} ends the conversation with {self.chatting_with}.")
        self.chatting_with = None
        self.chatting_buffer = []

    def conversation(self, target: str, content: str):
        WAIT = {'type': 'wait', 'arg1': None}
        if len(self.chatting_buffer) == 0 and (self.chatting_with is None or target is None): # set up the conversation
            self.logger.info(f"Setting up new conversation.")
            curr_events = self.curr_events
            curr_event = curr_events[-1] if len(curr_events) > 0 else None
            for event in curr_events:
                if event.event_type == "speech":
                    curr_event = event
                    break
            if curr_event is not None and curr_event.event_type == "speech":  # response to a conversation
                if target is None or target != curr_event.event_keywords[0]:
                    target = "someone"
                self.chatting_with = target
                self.chatting_buffer.append(
                    [self.curr_time, self.chatting_with, curr_event.event_description.split("] says: ")[1]])
            else:
                self.chatting_with = target
                self.chatting_buffer = []
        
        assert target == self.chatting_with, f"Target {target} is not equal to chatting_with {self.chatting_with}."

        curr_event = None
        for event in self.curr_events:
            if event.event_type == "speech":
                curr_event = event
                break
        if curr_event is not None and curr_event.event_type == "speech":  # response to a conversation
            self.logger.debug(f"response to a conversation : event_id is {curr_event.event_id}, description is {curr_event.event_description}")
            self.chatting_with = target
            self.chatting_buffer.append(
                [self.curr_time, self.chatting_with, curr_event.event_description.split("] says: ")[1]])
        
        self.logger.info(f"Chatting buffer length: {len(self.chatting_buffer)}")
        if len(self.chatting_buffer) > self.chat_time_limit:
            self.logger.info(f"Chatting with {self.chatting_with} for more than {self.chat_time_limit} seconds. Stop chatting.")
            self.end_conversation()
            return None

        if len(self.chatting_buffer) > 0 and self.chatting_buffer[-1][1] == self.name:
            if (self.curr_time - self.chatting_buffer[-1][0]).total_seconds() > 2:
                self.logger.info(f"{self.chatting_with} is not responding for more than 2 seconds. Stop chatting.")
                self.end_conversation()
                return None
            return WAIT
        
        if content in ["null", 'None']:
            self.logger.info(f"I want to stop the chatting.")
            self.end_conversation()
            return None
        
        self.chatting_buffer.append([self.curr_time + timedelta(seconds=1), (self.name, self.pose[:3]), content])
        self.logger.info(f"Final Chatting buffer length: {len(self.chatting_buffer)}")
    
        return {
            'type': 'remote_converse',
            'arg1': content,
            'arg2': 3200
        }

    def generate_captioning(self, prompt, img):
        if self.no_react:
            return "Do not revoke the llm in no react mode."
        response = self.generator.generate(prompt, img=img, json_mode=False)
        return response
    
    def delete_quotations(self, text):
        if isinstance(text, str):
            if text.startswith("\"") and text.endswith("\""):
                return text[1:-1]
            if text.startswith("'") and text.endswith("'"):
                return text[1:-1]
        return text

    def generate_utterance(self):
        prompt = open('agents/sentinel_challenge/meeting_prompts/coela_prompts/prompt_utterance.txt', 'r').read()
        task_description = open('agents/sentinel_challenge/meeting_prompts/task_description.txt', 'r').read()
        prompt = prompt.replace("$TaskDescription$", task_description)
        prompt = prompt.replace("$AgentList$", ", ".join(self.obs["agent_pos_dict"].keys()))
        prompt = prompt.replace("$Character$", self.get_character_description())

        prompt = prompt.replace("$Time$", self.curr_time.strftime("%H:%M:%S"))
        prompt = prompt.replace("$Place$", self.current_place if self.current_place is not None else self.meeting_place if self.meeting_place is not None and self.meeting_place in self.obs['accessible_places'] else f"open space: at {self.pose[:2]}")
        prompt = prompt.replace("$KnownPlaces$", self.get_nearest_places_description(self.get_meeting_target()))
        conversation_history_desp = '\n'.join([f"{chat[1][0]}: {chat[2]}" for chat in self.chatting_buffer[-4:]])
        prompt = prompt.replace("$Conversation_history$", self.get_conversation_description(20))
        prompt = prompt.replace("$Context$", self.describe_events(self.curr_events))
        self.logger.debug(f"Tracking place, current_place is {self.current_place}, meeting_place is {self.meeting_place}, accessible is {self.obs['accessible_places']}")
        self.logger.debug(f"Utterance prompt: {prompt}")
        response = self.delete_quotations(self.generator.generate(prompt, img=None, json_mode=False))
        self.logger.debug(f"Generated utterance: {response}")
        return response
    
    def generate_react_mode(self, curr_events, utterance):
        if utterance is None:
            prompt = open('agents/sentinel_challenge/meeting_prompts/coela_prompts/prompt_react_wo_chat.txt', 'r').read()
        else:
            prompt = open('agents/sentinel_challenge/meeting_prompts/coela_prompts/prompt_react.txt', 'r').read()
            prompt = prompt.replace("$Utterance$", utterance)
        task_description = open('agents/sentinel_challenge/meeting_prompts/task_description.txt', 'r').read()
        prompt = prompt.replace("$TaskDescription$", task_description)
        prompt = prompt.replace("$AgentList$", ", ".join(self.obs["agent_pos_dict"].keys()))

        prompt = prompt.replace("$Character$", self.get_character_description())
        prompt = prompt.replace("$Time$", self.curr_time.strftime("%H:%M:%S"))
        prompt = prompt.replace("$Place$", self.current_place if self.current_place is not None else self.meeting_place if self.meeting_place is not None and self.meeting_place in self.obs['accessible_places'] else f"open space: at {self.pose[:2]}")
        prompt = prompt.replace("$KnownPlaces$", self.get_nearest_places_description(self.get_meeting_target()))
        prompt = prompt.replace("$Context$", self.describe_events(curr_events))
        prompt = prompt.replace("$Conversation_history$", self.get_conversation_description(20))
        prompt = prompt.replace("$ActionHistory$", str(self.react_history[-10:]))
        self.logger.debug(f"Tracking place, current_place is {self.current_place}, meeting_place is {self.meeting_place}, accessible is {self.obs['accessible_places']}")
        self.logger.debug(f"React prompt: {prompt}")
        response = self.delete_quotations(self.generator.generate(prompt, img=None, json_mode=False))
        self.logger.debug(f"Generated react: {response}")
        if response is None:
            return "wait", None
        if utterance is not None and response.startswith("speak"):
            self.react_history.append("speak")
            return "speak", None
        self.react_history.append(response)
        if response.startswith("go to"):
            return "go", response.split("go to ")[1]
        if response.startswith("query"):
            return "query", response.split("query ")[1]
        if response == 'complete task':
            return 'complete task', None
        return "wait", None
    
    def get_places_description(self):
        places = []
        for place in self.s_mem.get_places():
            place_dict = self.s_mem.get_knowledge(place)
            if place_dict["coarse_type"] == "transit":
                continue
            places.append({"name": place, "type": place_dict["coarse_type"]})
        return json.dumps(places, indent=2)

    def describe_events(self, events):
        if events is None:
            return "No events."
        desc = ""
        for event in events:
            desc += f"type: {event.event_type}\ntime: {event.event_time}\nplace: {round_numericals(event.event_place)}\nkeywords: {event.event_keywords}\ncontent: {event.event_description}\n\n"
        return desc

    def get_curr_date(self):
        if self.curr_time is None:
            return None
        return self.curr_time.strftime("%A %B %d")

    def get_character_description(self):
        """EXAMPLE OUTPUT
           Name: Dolores Heitmiller
           Age: 28
           Innate traits: hard-edged, independent, loyal
           Learned traits: Dolores is a painter who wants live quietly and paint
             while enjoying her everyday life.
           Currently: Dolores is preparing for her first solo show. She mostly
             works from home.
           Lifestyle: Dolores goes to bed around 11pm, sleeps for 7 hours, eats
             dinner around 6pm.
            Groups:
           Daily plan requirement:
           Current Date: Monday, January 1
        """
        return f"""Name: {self.name}
Age: {self.scratch['age']}
Innate traits: {self.scratch['innate']}
Learned traits: {self.scratch['learned']}
Currently: {self.scratch['currently']}
Lifestyle: {self.scratch['lifestyle']}
Groups: {self.scratch['groups']}
Daily plan requirement: {self.scratch['daily_requirement']}
Held objects: {self.held_objects}
Cash: {self.obs['cash']}
Current date: {self.get_curr_date()}
"""