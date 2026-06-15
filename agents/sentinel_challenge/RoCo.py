import os
import traceback
from datetime import datetime, timedelta
import numpy as np

from agents.sentinel_challenge.base_nav import (
    Action,
    BaseNavigationMeetingAgent,
    NavAgentState,
    ThinkingModule,
)


class Decider(ThinkingModule):
    def __init__(self, generator, logger, name, ablate=""):
        super().__init__(generator, logger, name, type="roco", ablate=ablate)

    def rethink(self, curr_time, meeting_place, curr_eta, eta_history):
        prompt = open(os.path.join(self.prompt_path, "decide_rethink.txt"), "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        prompt = prompt.replace("$CurrentTime$", curr_time)
        prompt = prompt.replace("$SelfName$", self.name)
        prompt = prompt.replace("$CurrentPlace$", meeting_place)
        prompt = prompt.replace("$CurrentETA$", curr_eta)
        prompt = prompt.replace("$HistoricalETAs$", eta_history)
        self.logger.debug(f"planning_prompt: {prompt}")
        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            response_dict = self.parse_json(prompt, response)
            self.logger.debug(f"generated response: {response_dict}")
        except Exception as e:
            self.logger.error(
                f"Error deciding mode: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            response_dict = None
        return response_dict

    def start(self, agent_names, places, conversation_history):
        prompt = open(os.path.join(self.prompt_path, "decide_start.txt"), "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        prompt = prompt.replace("$SelfName$", self.name)
        prompt = prompt.replace("$AgentList$", agent_names)
        prompt = prompt.replace("$Places$", places)
        prompt = prompt.replace("$ConversationHistory$", conversation_history)
        self.logger.debug(f"planning_prompt: {prompt}")
        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            response_dict = self.parse_json(prompt, response)
            self.logger.debug(f"generated response: {response_dict}")
        except Exception as e:
            self.logger.error(
                f"Error extracting ETAs: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            response_dict = None
        return response_dict


class Discusser(ThinkingModule):
    def __init__(self, generator, logger, name, ablate=""):
        super().__init__(generator, logger, name, type="roco", ablate=ablate)

    def conclude_and_decide(self, curr_time, agent_names, places, conversation_history):
        prompt = open(os.path.join(self.prompt_path, "conclude_and_decide.txt"), "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        prompt = prompt.replace("$CurrentTime$", curr_time)
        prompt = prompt.replace("$SelfName$", self.name)
        prompt = prompt.replace("$AgentList$", agent_names)
        prompt = prompt.replace("$Places$", places)
        prompt = prompt.replace("$ConversationHistory$", conversation_history)
        self.logger.debug(f"planning_prompt: {prompt}")
        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            response_dict = self.parse_json(prompt, response)
            self.logger.debug(f"generated response: {response_dict}")
        except Exception as e:
            self.logger.error(
                f"Error concluding opinions: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            response_dict = None
        return response_dict

    def extract_info(self, agent_names, places, conversation_history):
        prompt = open(os.path.join(self.prompt_path, "extract_info.txt"), "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        prompt = prompt.replace("$SelfName$", self.name)
        prompt = prompt.replace("$AgentList$", agent_names)
        prompt = prompt.replace("$Places$", places)
        prompt = prompt.replace("$ConversationHistory$", conversation_history)
        self.logger.debug(f"planning_prompt: {prompt}")
        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            response_dict = self.parse_json(prompt, response)
            self.logger.debug(f"generated response: {response_dict}")
        except Exception as e:
            self.logger.error(
                f"Error extracting ETAs: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            response_dict = None
        return response_dict

    def analyze_and_plan(self, curr_time, pose, agent_opinions, places, conversation_history, known_poses, known_eta, known_sentinel_poses, stalling):
        prompt = open(os.path.join(self.prompt_path, "analyze_and_plan.txt"), "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        prompt = prompt.replace("$CurrentTime$", curr_time)
        prompt = prompt.replace("$SelfName$", self.name)
        prompt = prompt.replace("$SelfPose$", pose)
        prompt = prompt.replace("$AgentOpinions$", agent_opinions)
        prompt = prompt.replace("$Places$", places)
        prompt = prompt.replace("$ConversationHistory$", conversation_history)
        prompt = prompt.replace("$KnownPoses$", known_poses)
        prompt = prompt.replace("$KnownETA$", known_eta)
        prompt = prompt.replace("$KnownSentinelPoses$", known_sentinel_poses)
        if stalling:
            prompt = prompt.replace("$Stalling$", "The discussion has extended too long. Avoid throwing new questions and finalize as soon as possible!")
        else:
            prompt = prompt.replace("$Stalling$", '')
        self.logger.debug(f"planning_prompt: {prompt}")
        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            response_dict = self.parse_json(prompt, response)
            self.logger.debug(f"generated response: {response_dict}")
        except Exception as e:
            self.logger.error(
                f"Error extracting ETAs: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            response_dict = None
        return response_dict

    def speak(self, curr_time, pose, intent, agent_opinions, places, conversation_history, known_poses, known_eta, known_sentinel_poses, missing_info, stalling):
        prompt = open(os.path.join(self.prompt_path, "speak_speak.txt"), "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        prompt = prompt.replace("$CurrentTime$", curr_time)
        prompt = prompt.replace("$SelfName$", self.name)
        prompt = prompt.replace("$SelfPose$", pose)
        prompt = prompt.replace("$SpeechIntent$", intent)
        prompt = prompt.replace("$AgentOpinions$", agent_opinions)
        prompt = prompt.replace("$Places$", places)
        prompt = prompt.replace("$ConversationHistory$", conversation_history)
        prompt = prompt.replace("$KnownPoses$", known_poses)
        prompt = prompt.replace("$KnownETA$", known_eta)
        prompt = prompt.replace("$KnownSentinelPoses$", known_sentinel_poses)
        prompt = prompt.replace("$MissingInfo$", missing_info)
        if stalling:
            prompt = prompt.replace("$Stalling$", "The discussion has extended too long. Avoid throwing new questions and finalize as soon as possible!")
        else:
            prompt = prompt.replace("$Stalling$", '')
        self.logger.debug(f"planning_prompt: {prompt}")
        try:
            response = self.generator.generate(prompt, img=None, json_mode=False)
            self.logger.debug(f"generated response: {response}")
        except Exception as e:
            self.logger.error(
                f"Error extracting ETAs: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            response = None
        return response

    def query(self, curr_time, pose, intent, places):
        prompt = open(os.path.join(self.prompt_path, "query_action.txt"), "r").read()
        prompt = prompt.replace("$TaskDescription$", self.task_decription)
        prompt = prompt.replace("$CurrentTime$", curr_time)
        prompt = prompt.replace("$SelfName$", self.name)
        prompt = prompt.replace("$SelfPose$", pose)
        prompt = prompt.replace("$QueryIntent$", intent)
        prompt = prompt.replace("$Places$", places)
        self.logger.debug(f"planning_prompt: {prompt}")
        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            response_dict = self.parse_json(prompt, response)
            self.logger.debug(f"generated response: {response_dict}")
        except Exception as e:
            self.logger.error(
                f"Error generating query action: {e} with traceback: {traceback.format_exc()}. The response was {response}")
            response_dict = None
        return response_dict


class RoCoMeetingAgent(BaseNavigationMeetingAgent):
    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='openai', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0, init_generator=True,
                 detect_interval=-1, num_agents=1, enable_danger_zone=False):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger, lm_source, lm_id, max_tokens, temperature, top_p, init_generator, detect_interval, num_agents, enable_danger_zone)
        self.decider = Decider(generator=self.generator, logger=self.logger, name=self.name, ablate=self.ablate)
        self.discusser = Discusser(generator=self.generator, logger=self.logger, name=self.name, ablate=self.ablate)
        self.chat_time_limit = 60

    def reset(self, name, pose):
        super().reset(name, pose)

    def _process_obs(self, obs):
        super()._process_obs(obs)
        self.process_obs_with_sptial_knowledge(obs)

    def discuss_process_speech(self, obs):
        agent_names = ", ".join(obs["agent_pos_dict"].keys())
        places = self.get_nearest_places_description(self.get_meeting_target())
        current_message = self.get_conversation_description(limit=1)
        conversation_history = self.get_conversation_description()
        curr_time = self.curr_time.strftime('%H:%M:%S')
        extracted_info = {}

        if self.mode == NavAgentState.DISCUSS:
            if 'spatial_memory' in self.ablate or 'analyzer' in self.ablate:
                extracted_info = {}
            else:
                extracted_info = self.discusser.extract_info(agent_names=agent_names, places=places, conversation_history=current_message)
            conclusion_and_decision = self.discusser.conclude_and_decide(curr_time=curr_time, agent_names=agent_names, places=places, conversation_history=conversation_history)
            self.agent_opinions = conclusion_and_decision['agent_opinions']
            decision = conclusion_and_decision['agreement_check']
            if decision["agreed_location"] is not None:
                meeting_place = decision["agreed_location"]
                if meeting_place.startswith("<") and meeting_place.endswith(">"):
                    meeting_place = meeting_place[1:-1]
                self.meeting_place = meeting_place
            if decision["agreement_reached"] is True:
                self.enter_navigation_mode(goal_place=self.meeting_place)
        if 'ETA Map' in extracted_info:
            self.update_known_eta(extracted_info['ETA Map'])
        if 'Agent Poses' in extracted_info:
            self.update_known_poses(extracted_info['Agent Poses'])
        if 'Sentinel Poses' in extracted_info:
            self.update_known_sentinel_poses(extracted_info['Sentinel Poses'], shared=1)

    def discuss_act(self):
        places = self.get_nearest_places_description(self.get_meeting_target())
        conversation_history = self.get_conversation_description()
        curr_time = self.curr_time.strftime('%H:%M:%S')

        self.discussion_plan = self.discusser.analyze_and_plan(
            curr_time=curr_time, pose=self.get_outdoor_pose_description(),
            agent_opinions=self.get_agent_opinions_description(), places=places,
            conversation_history=conversation_history,
            known_poses=self.get_known_poses_description(),
            known_eta=self.get_known_eta_description(),
            known_sentinel_poses=self.get_known_sentinel_poses_description(),
            stalling=self.mode_time_counter > 30,
        )
        missing_info = "\n".join(self.discussion_plan['missing info'])
        action = {"type": "wait"}
        if self.discussion_plan["action"] == "wait":
            action = {"type": "wait"}
            self.discussion_plan = None
        elif self.discussion_plan["action"] == "query":
            operation = self.discusser.query(curr_time=curr_time, pose=self.get_outdoor_pose_description(), intent=self.discussion_plan['explanation'], places=places)
            if operation['type'] == 'query_nearby':
                operation['coordinate'][0], operation['coordinate'][1] = float(operation['coordinate'][0]), float(operation['coordinate'][1])
                operation['radius'] = float(operation['radius'])
                action = {'type': 'query_map_tool', 'arg1': 'query_nearby', 'arg2': operation['coordinate'], 'arg3': operation['radius']}
            elif operation['type'] == 'query_place':
                if operation['place'].startswith("<") and operation['place'].endswith(">"):
                    operation['place'] = operation['place'][1:-1]
                action = {'type': 'query_map_tool', 'arg1': 'query_place', 'arg2': operation['place']}
            elif operation['type'] == 'query_route':
                if operation['target'].startswith("<") and operation['target'].endswith(">"):
                    operation['target'] = operation['target'][1:-1]
                action = {'type': 'query_map_tool', 'arg1': 'query_route', 'arg2': operation['target']}
            else:
                raise NotImplementedError(f"operation query_map_tool type {operation['type']} is not supported")
            self.discussion_plan = None
        elif self.discussion_plan["action"] == "speak":
            intent = self.discussion_plan['explanation']
            speech = self.discusser.speak(curr_time=curr_time, pose=self.get_outdoor_pose_description(), intent=intent, agent_opinions=self.get_agent_opinions_description(), places=places, conversation_history=conversation_history, known_poses=self.get_known_poses_description(), known_eta=self.get_known_eta_description(), known_sentinel_poses=self.get_known_sentinel_poses_description(), missing_info=missing_info, stalling=self.mode_time_counter > 30)
            if speech == "null":
                action = {"type": "wait"}
            else:
                action = {"type": "remote_converse", "arg1": speech, "arg2": 3200}
            self.discussion_plan = None
        else:
            raise NotImplementedError(f"discussion plan type is not supported")
        return action

    def _act(self, obs):
        if self.banned:
            if self.pose[0] > -1000:
                return {"type": "teleport", "arg1": [-1500., -1500.]}
            return {"type": "task_complete"}
        self.logger.debug(f"Current mode is {self.mode}, while the trigger is {self.discussion_trigger}")
        action = None
        try:
            if self.mode is None:
                self.enter_discussion_mode(trigger="TASK START")
            if self.mode == NavAgentState.DISCUSS:
                self.mode_time_counter += 1
                if self.mode_time_counter > self.chat_time_limit:
                    if self.meeting_place is None:
                        self.logger.warning(f"Exceeding discussion limit but no agreed location. Terminating the task.")
                        action = {"type": "task_terminate"}
                        return action
                    else:
                        self.logger.warning(f"Exceeding discussion limit. Going to the most preferred location")
                        self.enter_navigation_mode(goal_place=self.meeting_place)
                        return {"type": "wait"}
                action = self.discuss_act()
            elif self.mode == NavAgentState.NAVIGATE:
                self.mode_time_counter += 1
                action, arrived = self.city_navigate(self.goal_place)
                if arrived:
                    action = {'type': 'task_complete'}
        except Exception as e:
            self.logger.error(f"Error in action generation: {e} with traceback: {traceback.format_exc()}. The plan was {action}")
            action = None
        self.action_history.append(Action(action, self.curr_time, self.curr_time))
        self.logger.debug(f"{self.name}'s current generated action is {action}.")
        assert action is None or isinstance(action, dict)
        self.last_action = action
        return self.last_action
