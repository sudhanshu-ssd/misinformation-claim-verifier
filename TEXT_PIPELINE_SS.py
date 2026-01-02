import pandas as pd
import faiss
import os
import json

def run_text_pipeline(claim: str, state: dict):
    
    retriever = state['retriever']
    reranker = state['reranker']
    classifier = state['classifier']
    summarizer = state['summarizer']
    fact_checker = state['fact_checker']
    df = state['df']
    evidence_corpus = state['evidence_corpus']
    faiss_index = state['faiss_index']

    try:
        retrieved_docs, indices = retriever.retrieve_evidence(claim, faiss_index, evidence_corpus)
    except Exception as e:
        print(f"Error retrieving evidence: {e}")
    try:
        reranked_docs = reranker.rerank_evidendce(claim, retrieved_docs)
    except Exception as e:
        print(f"Error reranking evidence: {e}")
    try:
        final_verdict, _ = classifier(claim, reranked_docs)
    except Exception as e:
        print(f"Error classifying claim: {e}")
    try:
        yoho ={}
        reranked_docs = [doc[1] for doc in reranked_docs]
        top_evidence_for_summary = reranked_docs[:3]
        _, explanation = summarizer(claim, top_evidence_for_summary, final_verdict)
        print("Summarization successful from RAG pipeline.")
    except Exception as e:
        print(f"Error summarizing evidence: {e}")
        print(reranked_docs[:1]) 
    try:
        sources_dict = {}
        if len(indices) > 0 and 'source' in df.columns and 'url' in df.columns:
            df_rel = df.iloc[indices]
            sources_dict = df_rel.groupby('source')['url'].first().to_dict()

        yoho = {
            "final_verdict": final_verdict,
            "explanation": explanation,
            "source": sources_dict
        }    
    
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        reranked_docs = []

    # if not reranked_docs:
        # Fallbacking to Google Fact Check 
        # print("No results from RAG, trying Google Fact Check...")
    result_web,result_fact_api,result_newsorg = fact_checker.check_claim(claim, reranker, classifier, summarizer)

    yolo =  {
        "final_verdict": result_web.get('verdict', 'NEUTRAL'),
        "explanation": result_web.get('summary', 'Could not verify claim.'),
        "sources": [
        {"source": s, "url": u}
            for s, u in zip(
                result_web.get('source', [] if isinstance(result_web.get('source'), list) else [result_web.get('source')]),
                result_web.get('URLs', [] if isinstance(result_web.get('URLs'), list) else [result_web.get('URLs')]))]}
    


    brook = {
        "verdict fro web": yolo.get('final_verdict', 'Cannot Verify from web analysis'),
        "verdict from google fact check": result_fact_api.get('verdict', 'NO verdict from Fact Check API'),
        "verdict from newsorg": result_newsorg.get('verdict', 'NO verdict from NewsOrg API'),
        "web explanation": (yolo.get('explanation', '') +("\n\n" + yoho.get('explanation') if yoho.get('explanation') else "")),
        "google fact check explanation": result_fact_api.get('summary', ''),
        "newsorg explanation": result_newsorg.get('summary', ''),
        "source From Web": {**yoho.get('source', {}), **{a['source']: a['url'] for a in yolo.get('sources', [])}},
        "URLs from Fact Check": result_fact_api.get('URLs', []),
        "URLs from NewsOrg": result_newsorg.get('URLs', []),
    }
    return brook