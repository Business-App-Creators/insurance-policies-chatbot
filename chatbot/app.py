"""Client demo for the Northline Assurance claims assistant.

    export OPENAI_API_KEY="your-key"
    python3 -m chatbot.app

Opens a local page. Each answer is the model's own words, then the quoted
manual text, then the citation. The quote is checked against the retrieved passage.
"""

import uuid

import gradio as gr

from chatbot.agents.sequential.graph import build_graph, format_result, run_config

GRAPH = build_graph()

EXAMPLES = [
    "When is a denial sent?",
    "When a site visit is required?",
    "What are the rules on payments to a minor?",
    "Where is the employee parking benefit described?",
]


def _status(result: dict) -> str:
    if not result.get("citations") and result.get("refusal"):
        return "Not in the manuals. No close passage was retrieved."
    if result.get("refusal"):
        return "Not in the manuals. The closest passage is shown below the answer."
    return "Answered from the manuals. The quote was checked against the retrieved text."


def ask(message: str, history: list[dict] | None, thread_id: str):
    message = (message or "").strip()
    if not message:
        raise gr.Error("Type a question first.")
    if not thread_id:
        thread_id = str(uuid.uuid4())
    prior = []
    for turn in history or []:
        content = turn.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        prior.append({"role": turn.get("role", "user"), "content": content})
    try:
        result = GRAPH.invoke(
            {"question": message, "history": prior, "attempts": 0},
            config=run_config(thread_id),
        )
    except Exception as exc:
        raise gr.Error(str(exc)[:300]) from exc
    updated = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": format_result(result)},
    ]
    return updated, "", _status(result), thread_id


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Northline Assurance") as demo:
        gr.Markdown(
            """
# Northline Assurance
### Claims policy assistant

A replica of a cited-answer agent. It searches the company's claims manuals, answers in its own words, and shows the exact passage it used. The quote is checked in code before you see it. The company and the manuals are fictional.
"""
        )
        thread_id = gr.State("")
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Ask the manuals", height=520)
                question = gr.Textbox(
                    placeholder="Ask how a claim is handled…",
                    show_label=False,
                    lines=2,
                )
                with gr.Row():
                    send = gr.Button("Ask", variant="primary")
                    clear = gr.Button("Clear")
                gr.Examples(examples=EXAMPLES, inputs=question)
            with gr.Column(scale=2):
                status = gr.Markdown("Ask a question to see whether the quote checked out.")
                gr.Markdown(
                    """
**What you are looking at**

The reply has three parts: the assistant's own words, the quoted manual text, and the citation (document, section, and file).

If the manuals do not decide the question, the reply says so and shows the closest passage instead of inventing a rule.
"""
                )
        send.click(ask, [question, chat, thread_id], [chat, question, status, thread_id])
        question.submit(ask, [question, chat, thread_id], [chat, question, status, thread_id])
        clear.click(lambda: ([], "", "Ask a question to see whether the quote checked out.", ""), outputs=[chat, question, status, thread_id])
    return demo


if __name__ == "__main__":
    build_demo().launch()
