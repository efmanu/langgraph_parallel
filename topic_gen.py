import operator
from typing import Annotated, TypedDict

from langchain_deepseek import ChatDeepSeek

from langgraph.types import Send
from langgraph.graph import END, StateGraph, START

from pydantic import BaseModel, Field


# Model and prompts
# Define model and prompts we will use
subjects_prompt = """Generate a comma separated list of between 2 and 5 examples related to: {topic}."""
summary_prompt = """Generate a summary of the following topic: {topic}"""
best_summary_prompt = """Below are a bunch of summaries about {topic}. Select the best one! Return the ID of the best one.

{summaries}"""

refine_summary_prompt = """Refine the following summary to make it more concise and informative: {summary}."""


class Subjects(BaseModel):
    subjects: list[str]


class Summary(BaseModel):
    summary: str


class BestSummary(BaseModel):
    id: int = Field(description="Index of the best summary, starting with 0", ge=0)


model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=1000,
    timeout=None,
    max_retries=2,
)

# Graph components: define the components that will make up the graph


# This will be the overall state of the main graph.
# It will contain a topic (which we expect the user to provide)
# and then will generate a list of subjects, and then a summary for
# each subject
class OverallState(TypedDict):
    topic: str
    subjects: list
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    summaries: Annotated[list, operator.add]
    best_selected_summary: str


# This will be the state of the node that we will "map" all
# subjects to in order to generate a summary
class SummaryState(TypedDict):
    subject: str
    summary: Annotated[list, operator.add]


# This is the function we will use to generate the subjects of the summaries
def generate_topics(state: OverallState) -> OverallState:
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}


# Here we generate a summary, given a subject
def generate_summary(state: SummaryState) -> SummaryState:
    prompt = summary_prompt.format(topic=state["subject"])
    response = model.with_structured_output(Summary).invoke(prompt)
    return {"summary": [response.summary]}

def refine_summary(state: SummaryState) -> OverallState:
    prompt = refine_summary_prompt.format(summary=state["summary"])
    response = model.with_structured_output(Summary).invoke(prompt)
    return {"summaries": [response.summary]}

# Here we define the logic to map out over the generated subjects
# We will use this an edge in the graph
def continue_to_summaries(state: OverallState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [Send("generate_summary", {"subject": s}) for s in state["subjects"]]


# Here we will judge the best summary
def best_summary(state: OverallState) -> OverallState:
    summaries = "\n\n".join(state["summaries"])
    prompt = best_summary_prompt.format(topic=state["topic"], summaries=summaries)
    response = model.with_structured_output(BestSummary).invoke(prompt)
    return {"best_selected_summary": state["summaries"][response.id]}

def continue_to_refine(state: SummaryState):
    return Send("refine_summary", state)

def compile_graph():
    # Construct the graph: here we put everything together to construct our graph
    graph = StateGraph(OverallState)
    graph.add_node("generate_topics", generate_topics)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("refine_summary", refine_summary)
    graph.add_node("best_summary", best_summary)
    graph.add_edge(START, "generate_topics")
    graph.add_conditional_edges("generate_topics", continue_to_summaries, ["generate_summary"])
    graph.add_conditional_edges("generate_summary", continue_to_refine, ["refine_summary"])
    graph.add_edge("refine_summary", "best_summary")
    graph.add_edge("best_summary", END)
    app = graph.compile()
    return app

app = compile_graph()

png_image = app.get_graph().draw_mermaid_png()

with open("langgraph_diagram_n.png", "wb") as f:
    f.write(png_image)


state = app.invoke({"topic": "birds"})

print(state)