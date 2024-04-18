from abc import ABC, abstractmethod


class Verifier(ABC):
    @abstractmethod
    def verify(self, response: str, data: dict | None = None) -> (bool, str):
        raise NotImplementedError("You must implement `verify` method.")
