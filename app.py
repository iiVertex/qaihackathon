import gradio as gr

with gr.Blocks(title="Review Inspector") as demo:
    gr.Markdown("# Review Inspector\n\nHello world: Space skeleton is running.")

if __name__ == "__main__":
    demo.launch()
