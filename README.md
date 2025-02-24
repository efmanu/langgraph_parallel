# Building Parallel Workflows with LangGraph: A Practical Guide
## Introduction
Large Language Models (LLMs) have become powerful tools for natural language processing, but orchestrating complex workflows with them can be challenging. This is where LangGraph enters the picture - a specialized framework designed to bring structure and efficiency to LLM applications.
LangGraph stands out by treating your LLM application as a directed graph, where each node represents a specific operation and edges define the flow of data. What makes it particularly powerful is its ability to handle parallel processing, allowing multiple operations to execute simultaneously when they don't depend on each other's outputs.
In this guide, we'll explore a practical example: building a topic-based summary generator. Imagine you want to generate comprehensive summaries about a topic, but instead of a linear approach, you want to explore multiple angles simultaneously. Our application will take a topic (like "birds"), generate several subtopics in parallel, create summaries for each, and then select the best one - all while maintaining clean state management and type safety.
This real-world use case demonstrates how LangGraph's parallel processing capabilities can significantly improve efficiency and create more sophisticated LLM applications. Let's dive into how we can build this system step by step.
## Understanding the Building Blocks
LangGraph's power lies in its core components that work together to create structured LLM applications. At its heart is the StateGraph, which serves as the main container for your application's logic. Each operation in your workflow is represented by a Node, which can process data independently and in parallel when possible.
The flow of data between nodes is managed through TypedDict-based states, ensuring type safety and clear data contracts. These states can be annotated with operators like operator.add to handle parallel processing results. Edges connect these nodes, defining both sequential and conditional paths through your application.
This structured approach allows us to build complex, parallel workflows while maintaining clarity and reliability in our LLM applications.
##State Management
LangGraph uses TypedDict to define structured state, ensuring clarity and type safety. Annotations, particularly Annotated[list, operator.add], play a crucial role in merging multiple outputs into a single list, enabling efficient parallel processing. Best practices for state design include:
Keeping state definitions minimal yet expressive
Using clear type annotations for maintainability
Ensuring immutability where feasible
Structuring state logically to support parallelism and conditional flows

## Core Components
### Nodes and Edge Design
In LangGraph, nodes are the fundamental processing units that follow the single-responsibility principle. Each node has a clear, specific task - whether it's generating topics, creating summaries, or selecting the best content.
### State Management
State transitions are handled through TypedDict classes that define clear contracts for data flow. The OverallState maintains the global context, while SummaryState manages data for individual parallel operations.
### Parallel Flow Configuration
Edges define how data flows between nodes. Using add_conditional_edges, we can create parallel processing paths. For example:
```python
graph.add_conditional_edges(
    "generate_topics", 
    continue_to_summaries, 
    ["generate_summary"]
)
```
This setup allows multiple summary generations to run concurrently, improving efficiency.
## Example Implementation: LangGraph Parallel Workflow Processing
Our implementation demonstrates parallel processing by building a topic summarization system that:
- Takes a topic like "birds"
- Generates multiple subtopics (e.g., "migration patterns", "nesting habits", "feeding behaviors")
- Creates summaries for each subtopic simultaneously
- Refines and selects the best summary

The parallel approach allows us to process multiple summaries concurrently, significantly reducing overall execution time compared to sequential processing.
### Setting Up the Environment
To build our parallel processing application, we'll need to set up our environment with key dependencies:
```bash
pip install langgraph langchain-deepseek pydantic typing-extensions
```
We configure DeepSeek as our LLM with specific parameters:
```python
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=1000,
    timeout=None,
    max_retries=2
)
```
This setup provides the foundation for our parallel processing workflow, ensuring reliable and consistent responses from our LLM.
### Defining the Data Models
LangGraph applications require well-structured data models to ensure reliable processing and type safety. Our implementation uses three key components:
#### Pydantic Models for Structured Output
```python
class Subjects(BaseModel):
    subjects: list[str]

class Summary(BaseModel):
    summary: str

class BestSummary(BaseModel):
    id: int = Field(description="Index of the best summary, starting with 0", ge=0)
```
These Pydantic models ensure that LLM outputs are properly structured and validated. They act as contracts between the LLM and our application, preventing unexpected data formats.
#### State Definitions
```python
class OverallState(TypedDict):
  topic: str
  subjects: list
  summaries: Annotated[list, operator.add]
  best_selected_summary: str

class SummaryState(TypedDict):
    subject: str
    summary: Annotated[list, operator.add]
```
The state definitions serve two crucial purposes:
- Managing the overall workflow state (`OverallState`)
- Handling individual parallel processing states (`TopicState`)
### Type Safety Considerations
- Use of TypedDict ensures compile-time type checking
- Annotated with operator.add enables proper merging of parallel outputs
- Field descriptions and validators (like ge=0) prevent invalid data
- Clear separation between global and node-specific states prevents state pollution
This robust type system helps catch errors early and makes the codebase more maintainable and self-documenting.
This section explains the core data modeling aspects of your LangGraph application, emphasizing type safety and state management through Pydantic and TypedDict classes. It shows how proper data modeling supports parallel processing while maintaining code reliability.
##. Building Processing Nodes
In our LangGraph application, nodes are the fundamental processing units. Each node has a specific responsibility and can be executed independently or in parallel when possible.
#### Topic Generation Node
This node takes the initial topic and gmultople related subjects for exploration.
```python
def generate_topics(state: OverallState) -> OverallState:
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}
```
Key Features:
- Uses a formatted prompt to generate related subjects
- Returns subjects for parallel processing
- Structured output ensures consistent data format
#### Parallel Summary Generation Node
This node processes individual subjects in parallel to generate detailed summaries.
```python   
def generate_summary(state: SummaryState) -> SummaryState:
    prompt = summary_prompt.format(topic=state["subject"])
    response = model.with_structured_output(Summary).invoke(prompt)
    return {"summary": [response.summary]}
```
Key Features:
- Processes a single subject independently
- Can be executed in parallel for multiple subjects
- Returns structured output for consistency
#### Summary Refinement Node
This node takes generated summaries and refines them to be more interesting and detailed.
```python
def refine_summary(state: SummaryState) -> OverallState:
    prompt = refine_summary_prompt.format(summary=state["summary"])
    response = model.with_structured_output(Summary).invoke(prompt)
    return {"summaries": [response.summary]}
```
Key Features:
- Improves the quality of generated summaries
- Maintains consistent state structure
- Processes refined content independently
- Converts SummaryState to OverallState for final processing
#### Best Summary Selection Node
This node evaluates all refined summaries and selects the most suitable one.
```python
def best_summary(state: OverallState) -> OverallState:
  summaries = "\n\n".join(state["summaries"])
  prompt = best_summary_prompt.format(topic=state["topic"], summaries=summaries)
  response = model.with_structured_output(BestSummary).invoke(prompt)
  return {"best_selected_summary": state["summaries"][response.id]}
``` 
Key Features:
- Combines all summaries for comparison
- Uses structured output for consistent selection
- Returns the index of the best summary
- Updates the final state with the selected summary
#### Node Connection Logic
The nodes are connected using conditional edges that enable parallel processing:
```python
def continue_to_summaries(state: OverallState):
  return [Send("generate_summary", {"subject": s}) for s in state["subjects"]]

def continue_to_refine(state: SummaryState):
  return Send("refine_summary", state)
```
Key Features:
- continue_to_summaries: Maps subjects to parallel summary generation tasks
- continue_to_refine: Directs refined summaries to the final selection
- Uses Send objects to manage state transitions
- Enables parallel processing through proper state mapping
#### Prompts Used
The nodes use carefully crafted prompts to guide the LLM:
```python
subjects_prompt = """Generate a comma separated list of between 2 and 5 examples related to: {topic}."""

summary_prompt = """Generate a summary of the following topic: {topic}"""

best_summary_prompt = """Below are a bunch of summaries about {topic}. Select the best one! Return the ID of the best one.
{summaries}"""

refine_summary_prompt = """Refine the following summary to make it more concise and informative: {summary}."""
```
This node architecture enables:
1. Parallel processing of multiple subjects
2. Clear separation of concerns
3. Type-safe state transitions
4. Efficient workflow management
Each node is designed to be independent and stateless, making the system more maintainable and easier to debug. The parallel processing capabilities significantly reduce the overall execution time compared to a sequential approach.
### Orchestrating the Workflow
The workflow orchestration in LangGraph involves constructing a directed graph, configuring edges for parallel processing, and setting up the execution flow. Let's break down each component:
#### Graph Construction
The graph is constructed using the StateGraph class and compiled into an executable application:
```python
def compile_graph():
  # Initialize the graph with our OverallState type
  graph = StateGraph(OverallState)
  # Add processing nodes
  graph.add_node("generate_topics", generate_topics)
  graph.add_node("generate_summary", generate_summary)
  graph.add_node("refine_summary", refine_summary)
  graph.add_node("best_summary", best_summary)
  # Define the sequential flow
  graph.add_edge(START, "generate_topics")
  graph.add_edge("best_summary", END)
  # Add conditional edges for parallel processing
  graph.add_conditional_edges(
    "generate_topics",
    continue_to_summaries,
    ["generate_summary"]
  )
  graph.add_conditional_edges(
    "generate_summary",
    continue_to_refine,
    ["refine_summary"]
  )
  graph.add_edge("refine_summary", "best_summary")
  # Compile the graph into an executable
  return graph.compile()
```
#### Conditional Edge Configuration
The conditional edges enable parallel processing by dynamically creating paths based on the state.
#### Graph Execution Flow
The workflow follows this sequence:
1. Start → generate_topics
2. generate_topics → multiple generate_summary instances (parallel)
3. generate_summary → refine_summary
4. refine_summary → best_summary
5. best_summary → End
5. Visualization and Debugging
LangGraph provides powerful tools for visualizing workflows, inspecting states, and debugging parallel processes. Let's explore these capabilities in detail.
#### Basic Graph Visualization
LangGraph can generate Mermaid diagrams to visualize your workflow:
```python
#Generate and save the graph visualization
app = compile_graph()
png_image = app.get_graph().draw_mermaid_png()
#Save to file
with open("langgraph_diagram.png", "wb") as f:
  f.write(png_image)
```
#### Execution and Runtime Behavior
##### Basic Execution
The simplest way to execute the LangGraph workflow is through the invoke method:
```python
#Compile the graph
app = compile_graph()
#Execute with a single topic
state = app.invoke({"topic": "birds"})
print(state)
```
Expected output structure:

{'topic': 'birds', 'subjects': ['sparrow', 'eagle', 'penguin', 'parrot'], 'summaries': ['Sparrows, small passerine birds of the family Passeridae, are adaptable and social, thriving in urban and rural areas worldwide. They primarily feed on seeds and insects, contributing to ecosystems by controlling pests and dispersing seeds. Known for their distinctive chirping, sparrows often form flocks. However, habitat loss and pollution threaten some species, causing population declines in certain regions.', 'Eagles are large birds of prey in the Accipitridae family, known for their powerful build, keen eyesight, and role as apex predators. Found on every continent except Antarctica, they inhabit diverse environments like forests, mountains, and plains. With over 60 species, including the bald and golden eagles, they are cultural symbols of power and freedom, featured in mythology, folklore, and national emblems.', 'Penguins are aquatic, flightless birds primarily found in the Southern Hemisphere, especially Antarctica. Adapted for swimming, they have countershaded plumage and flippers. Penguins feed on krill, fish, and squid, spending half their lives on land and half in water. They live in large colonies and exhibit unique behaviors like waddling and tobogganing. Species range from the small blue penguin to the large emperor penguin. Threats include climate change, overfishing, and pollution.', 'Parrots are colorful, intelligent birds in the order Psittaciformes, found in tropical and subtropical regions. Known for mimicking sounds, including human speech, they have strong, curved beaks and zygodactyl feet, making them excellent climbers. Social and often seen in flocks, parrots have a varied diet of seeds, fruits, and nuts. Popular as pets for their vibrant colors and interactive nature, they require significant care. Conservation efforts are crucial due to threats from habitat destruction and the pet trade.'], 'best_selected_summary': 'Sparrows, small passerine birds of the family Passeridae, are adaptable and social, thriving in urban and rural areas worldwide. They primarily feed on seeds and insects, contributing to ecosystems by controlling pests and dispersing seeds. Known for their distinctive chirping, sparrows often form flocks. However, habitat loss and pollution threaten some species, causing population declines in certain regions.'}

### Complete Code:
```python
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
```
### Conclusion
LangGraph provides a powerful framework for building parallel processing applications with LLMs, as demonstrated through our topic summarization example. Key concepts include state management through TypedDict, parallel processing with conditional edges, and robust error handling. Best practices we've covered include maintaining clear node responsibilities, implementing proper state validation, and utilizing visualization tools for debugging. The framework's ability to handle parallel operations while maintaining type safety and state consistency makes it an excellent choice for complex LLM applications. For next steps, developers should explore advanced features like custom node implementations, integration with different LLM providers, and implementing more complex parallel processing patterns. Valuable resources include the official LangGraph documentation, LangChain's integration guides, and the growing community of LangGraph developers sharing implementations and best practices. As LLM applications continue to evolve, LangGraph's structured approach to workflow management will become increasingly valuable for building scalable and maintainable applications.
### References
- LangGraph Official documentation
- Github Repo: https://github.com/langchain-ai/langgraph
- LangChain Integration: https://python.langchain.com/docs/integrations/langgraph/
- LangGraph Community: https://github.com/langchain-ai/langgraph/discussions
- LangGraph Examples: https://github.com/langchain-ai/langgraph/tree/main/examples