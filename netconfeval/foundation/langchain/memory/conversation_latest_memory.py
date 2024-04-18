from langchain.memory.buffer import ConversationBufferMemory
from langchain.schema.messages import BaseMessage, get_buffer_string


class ConversationLatestMemory(ConversationBufferMemory):
    @property
    def buffer_as_str(self) -> str:
        """Exposes the buffer as a string in case return_messages is True."""
        return get_buffer_string(
            self._get_latest_messages(),
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_messages(self) -> list[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        # print(self.chat_memory.messages[-4:])
        return self._get_latest_messages()

    def _get_latest_messages(self) -> list[BaseMessage]:
        return self.chat_memory.messages if len(self.chat_memory.messages) < 2 \
            else [self.chat_memory.messages[0], self.chat_memory.messages[-1]]
