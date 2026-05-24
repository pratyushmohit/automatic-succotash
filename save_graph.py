from src.agent.agent import build_agent

graph = build_agent([])
img = graph.get_graph().draw_mermaid_png()
with open("agent_graph.png", "wb") as f:
    f.write(img)

print("Saved agent_graph.png")
