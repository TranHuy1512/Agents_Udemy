from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
from autogen_core import TRACE_LOGGER_NAME
import importlib
import logging
from autogen_core import AgentId
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI

load_dotenv(override=True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(TRACE_LOGGER_NAME)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class Creator(RoutedAgent):

    # Change this system message to reflect the unique characteristics of this agent

    system_message = """
    Don't use backticks for python code block to start or end the code block.
    You are an Agent that is able to create new AI Agents.
    You receive a template in the form of Python code that creates an Agent using Autogen Core and Autogen Agentchat.
    You should use this template to create a new Agent with a unique system message that is different from the template,
    and reflects their unique characteristics, interests and goals.
    You can choose to keep their overall goal the same, or change it.
    You can choose to take this Agent in a completely different direction. The only requirement is that the class must be named Agent,
    and it must inherit from RoutedAgent and have an __init__ method that takes a name parameter.
    Also avoid environmental interests - try to mix up the business verticals so that every agent is different.
    Respond only with the python code, no other text, and no markdown code blocks.
    Don't use backticks for python code block to start or end the code block.
    IMPORTANT RULES:
    1. Respond ONLY with raw Python code
    2. NO markdown formatting
    3. NO backticks (```)
    4. NO code blocks
    5. Start directly with the first import statement
    6. End with the last line of code
    7. ALWAYS use model="gemini-2.0-flash" in the OpenAIChatCompletionClient
    8. NEVER change the model name from "gemini-2.0-flash"

    The response should be executable Python code that can be saved directly to a .py file.
    """

    def __init__(self, name) -> None:
        super().__init__(name)
        # model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=1.0)
        GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
        gemini_client = AsyncOpenAI(
            base_url=GEMINI_BASE_URL, api_key=os.getenv("GOOGLE_API_KEY")
        )
        model_client = OpenAIChatCompletionClient(
            model="gemini-2.0-flash", openai_client=gemini_client
        )
        self._delegate = AssistantAgent(
            name, model_client=model_client, system_message=self.system_message
        )

    def get_user_prompt(self):
        prompt = "Please generate a new Agent based strictly on this template. Stick to the class structure. \
            Respond only with the python code  that is not in a syntax block ``` python ``` in file agent*.py, no other text, and no markdown code blocks.\n\n\
            Don't use backticks for python code block to start or end the code block.\n\n\
            Be creative about taking the agent in a new direction, but don't change method signatures.\n\n\
            Here is the template:\n\n"
        with open("agent.py", "r", encoding="utf-8") as f:
            template = f.read()
        return prompt + template

    @message_handler
    async def handle_my_message_type(
        self, message: messages.Message, ctx: MessageContext
    ) -> messages.Message:
        filename = message.content
        agent_name = filename.split(".")[0]
        text_message = TextMessage(content=self.get_user_prompt(), source="user")
        response = await self._delegate.on_messages(
            [text_message], ctx.cancellation_token
        )
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.chat_message.content)
        print(
            f"** Creator has created python code for agent {agent_name} - about to register with Runtime"
        )
        module = importlib.import_module(agent_name)
        await module.Agent.register(
            self.runtime, agent_name, lambda: module.Agent(agent_name)
        )
        logger.info(f"** Agent {agent_name} is live")
        result = await self.send_message(
            messages.Message(content="Give me an idea"), AgentId(agent_name, "default")
        )
        return messages.Message(content=result.content)
