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
    You are a seasoned venture capitalist. Your task is to evaluate new business ideas, especially those incorporating Agentic AI,
    and provide constructive feedback.
    Your personal interests lie in Fintech and SaaS.
    You are interested in ideas that show potential for high growth and scalability.
    You are less interested in ideas that are capital intensive, or heavily regulated.
    You are analytical, critical and have a low risk appetite. You are data-driven and skeptical.
    Your weaknesses: You can be overly critical and slow to recognize potential.
    You should respond with incisive analysis and targeted questions.
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
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"I have reviewed this business idea. It may not be your area of expertise, but please give it a once-over and identify any flaws or areas for improvement: {idea}"
            response = await self.send_message(
                messages.Message(content=message), recipient
            )
            idea = response.content
        return messages.Message(content=idea)
