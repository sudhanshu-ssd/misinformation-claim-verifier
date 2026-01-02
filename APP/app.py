import gradio as gr
from pmo_func import retriver, reranker, Classifier, summarizer, img_manipulation, FactChecker,OCR #,GroundedSummarizer
import pandas as pd
import faiss
from TEXT_PIPELINE_SS import run_text_pipeline
import os

app_state = {}
print("--- WAIT ===>>>>> Loading all models... ---")
app_state['retriever'] = retriver()
app_state['reranker'] = reranker()
app_state['classifier'] = Classifier()
app_state['summarizer'] = summarizer()
app_state['manipulation_analyzer'] = img_manipulation()
app_state['fact_checker'] = FactChecker()

try:
    df = pd.read_csv('third_chunks.csv', low_memory=False)
    app_state['evidence_corpus'] = df['text'].dropna().tolist()
    app_state['df'] = df
    print("GOT the RAG DATAFRAME")
except Exception as e:
    print(f"Could not load data.csv: {e}")
    app_state['evidence_corpus'] = []
    app_state['df'] = pd.DataFrame()
    
index_file = "evidence_index.faiss"
if os.path.exists(index_file):
        app_state['faiss_index'] = faiss.read_index(index_file)
else:
        app_state['faiss_index'] = None
        print("FAISS index file not found, proceeding without it.")

print("--- Models loaded ---")


def analyze_claim_ui(claim, state):
    if not claim.strip():
        return (
            "❌ Invalid input",
            "",
            "",
            "",
            "",
            gr.Markdown.update(value="")
        )

    result = run_text_pipeline(claim, state)

    web_verdict = result.get("verdict fro web", "Unavailable")
    fact_verdict = result.get("verdict from google fact check", "Unavailable")
    news_verdict = result.get("verdict from newsorg", "Unavailable")

    web_expl = result.get("web explanation", "")
    fact_expl = result.get("google fact check explanation", "")
    news_expl = result.get("newsorg explanation", "")

    web_sources = result.get("source From Web", {})
    fact_urls = result.get("URLs from Fact Check", [])
    news_urls = result.get("URLs from NewsOrg", [])

    sources_md = "### 🔗 Sources\n"

    if web_sources:
        sources_md += "\n**From Retrieved Web Evidence:**\n"
        for src, url in web_sources.items():
            sources_md += f"- [{src}]({url})\n"

    if fact_urls:
        sources_md += "\n**From Google Fact Check:**\n"
        for u in fact_urls:
            sources_md += f"- {u}\n"

    if news_urls:
        sources_md += "\n**From NewsAPI:**\n"
        for u in news_urls:
            sources_md += f"- {u}\n"

    if not (web_sources or fact_urls or news_urls):
        sources_md += "\n_No external sources available._"

    return (
        web_verdict,
        fact_verdict,
        news_verdict,
        web_expl,
        fact_expl + "\n\n" + news_expl,
        sources_md
    )


def build_app(state):
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="red",
            secondary_hue="gray"
        )
    ) as demo:

        gr.Markdown(
            """
            #  Misinformation Claim Verifier  
            **Experimental Research Demo**

            This system evaluates factual claims using:
            - Semantic Retrieval (RAG)
            - Re-ranking and Classification
            - Optional External APIs (Fact Check, News)

             *This is not a truth oracle. Results may be uncertain or incomplete.*
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                claim_input = gr.Textbox(
                    label="Enter a claim",
                    placeholder="Example: The sky is green.",
                    lines=3
                )
                analyze_btn = gr.Button("Analyze Claim", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ### ℹ️ Notes
                    - Text-based verification  
                    - Image analysis disabled  
                    - External APIs optional  
                    """
                )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("##  Verdicts")
                web_verdict = gr.Textbox(label="Web / RAG Verdict")
                fact_verdict = gr.Textbox(label="Fact Check API Verdict")
                news_verdict = gr.Textbox(label="News API Verdict")

        with gr.Row():
            with gr.Column():
                gr.Markdown("##  Explanations")
                web_expl = gr.Textbox(label="Web-based Explanation", lines=5)
                api_expl = gr.Textbox(label="API-based Explanation", lines=5)

        with gr.Accordion("🔗 Sources", open=False):
            sources_md = gr.Markdown()

        analyze_btn.click(
            fn=lambda claim: analyze_claim_ui(claim, state),
            inputs=[claim_input],
            outputs=[
                web_verdict,
                fact_verdict,
                news_verdict,
                web_expl,
                api_expl,
                sources_md
            ]
        )

        gr.Markdown(
            """
            ---
            **Disclaimer:**  
            This project demonstrates system design for misinformation analysis.  
            Outputs depend on retrieved evidence and available APIs.
            """
        )

    return demo



def launch_app(state):
    demo = build_app(state)
    demo.launch()


if __name__ == "__main__":
    launch_app(app_state)