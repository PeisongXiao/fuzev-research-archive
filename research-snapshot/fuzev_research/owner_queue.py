"""Submit native work to the engine's executor."""


class OwnerQueue:
    def __init__(self, executor):
        self.executor = executor

    def submit(self, function):
        return self.executor.submit(function)
