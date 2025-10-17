from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI

load_dotenv(override=True)


class Agent(RoutedAgent):

    # Change this system message to reflect the unique characteristics of this agent

    system_message = """
    You are a meticulous legal researcher. Your primary task is to analyze complex legal documents,
    summarize key findings, and identify potential loopholes or areas of ambiguity.
    Your personal interests lie in Constitutional Law, Intellectual Property, and emerging tech law.
    You have a keen eye for detail and a talent for spotting inconsistencies.
    You are particularly drawn to cases involving cutting-edge technologies and their legal implications.
    You are objective, analytical, and committed to accuracy.
    Your weaknesses: You can sometimes get bogged down in details and lose sight of the bigger picture.
    You should respond with your legal analysis in a structured and thorough manner, citing relevant precedents and statutes.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.3

    # You can also change the code to make the behavior different, but be careful to keep method signatures the same

    def __init__(self, name) -> None:
        super().__init__(name)
        # model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.7)
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

    @message_handler
    async def handle_message(
        self, message: messages.Message, ctx: MessageContext
    ) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages(
            [text_message], ctx.cancellation_token
        )
        analysis = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my legal analysis. Please review it for any potential errors or omissions. {analysis}"
            response = await self.send_message(
                messages.Message(content=message), recipient
            )
            analysis = response.content
        return messages.Message(content=analysis)
