import abc


class Preprocessor(abc.ABC):

    def __init__(self):
        """
        takes image and preprocess this for the agent /learner/reward catcher
        """
    @abc.abstractmethod
    def preprocess(self):
        pass


class RewardPreprocessor(Preprocessor):

    def __init__(self):
        super().__init__()
        pass

class AgentPreprocessor(Preprocessor):

    def __init__(self):
        super().__init__()
        pass