from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]

llm = ChatOpenAI(temperature=0, model="gpt-4")

react_prompt = hub.pull("hwchase17/react")

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)


react_prompt_with_format_instructions = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(llm, tools, prompt=react_prompt_with_format_instructions)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

chain = agent_executor


def main():
    print("Hello from langchain-course!")
    result = chain.invoke(
        input={
            "input": "search for 3 job posting in linked for solutions architect in muscat, Oman and list thier details",
        }
    )
    print(result)

    if "error" in result:
        print("Error:", result["error"])
    else:
        print("Input:", output_parser.parse(result["input"]))
        print("Answer:", output_parser.parse(result["output"]).answer)
        print("Sources:")
        for source in output_parser.parse(result["output"]).sources:
            print("-", source.url)


if __name__ == "__main__":
    main()
