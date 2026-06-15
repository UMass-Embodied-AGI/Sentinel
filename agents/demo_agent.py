import copy

class DemoAgent:
    def __init__(self, action_list):
        self.action_buf = copy.deepcopy(action_list)

    def reset(self, action_list):
        self.action_buf = copy.deepcopy(action_list)

    def act(self, obs):
        if obs['action_status'] == "ONGOING":
            return None
        return self.action_buf.pop(0) if len(self.action_buf) > 0 else None


class DemoAgentProcess:
    """Thin wrapper around DemoAgent that mirrors the AgentProcess interface used in challenge.py."""

    def __init__(self, name: str, action_list: list):
        self.name = name
        self._agent = DemoAgent(action_list)
        self._pending_obs = None

    def update(self, obs):
        self._pending_obs = obs

    def act(self):
        return self._agent.act(self._pending_obs)

    def request_chat(self, content):
        pass

    def get_utterance(self, steps):
        return None

    def start(self):
        pass  # no subprocess needed
