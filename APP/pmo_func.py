import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import pipeline
from PIL import Image, ImageChops, ImageEnhance
import torch
from google.cloud import vision
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import trafilatura as tra

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class retriver:
    def __init__(self):
        self.retrivermodel = SentenceTransformer('all-MiniLM-L6-v2')

    def build_faiss_idx(self, evidence_corpus):
        embeddings = self.retrivermodel.encode(evidence_corpus)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))
        faiss.write_index(index, "evidence_index.faiss")
        return index

    def retrieve_evidence(self, claim, index, evidence_corpus, top_k=10):
        claim_embedding = self.retrivermodel.encode([claim])
        distances, indices = index.search(np.array(claim_embedding, dtype=np.float32), top_k)
        retrieved_docs = [evidence_corpus[i] for i in indices[0]]
        return retrieved_docs, indices[0]

class reranker:
    def __init__(self):
        try:
            self.reranker_model = CrossEncoder(
                'cross-encoder/ms-marco-MiniLM-L-6-v2',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("Got the reranker")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(" Reranker too big for GPU --> using CPU ")
                self.reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
            else:
                raise e
    def rerank_evidendce(self,claim, evidence_list):
        sentance_pairs= [[claim,evidence] for evidence in evidence_list]
        score= self.reranker_model.predict(sentance_pairs)
        scored_evidence= sorted(zip(score, evidence_list), reverse=True)
        return scored_evidence

class Classifier:
    def __init__(self):
        self.model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        self.label_names = ["entailment", "neutral", "contradiction"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Classifier device:", self.device)

        try:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                dtype=dtype,
                low_cpu_mem_usage=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except RuntimeError as e:
            if "CUDA out of memory." in str(e):
                print(" Classifier too big for GPU --> using CPU ")
                self.device = torch.device("cpu")
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            else:
                raise RuntimeError(f"Could not fetch model from Hugging Face | {e}")

    def classify(self, claim, top_evidence):
        self.verdicts = []
        evidences = [e[1] for e in top_evidence]
        
        if not evidences:
            raise ValueError("No evidence provided")
        
        try:
            batch_size = 2  
            for i in range(0, len(evidences), batch_size):
                batch_evidences = evidences[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch_evidences,
                    [claim] * len(batch_evidences),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                )
                
                with torch.no_grad():
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                
                probs = torch.softmax(outputs.logits, dim=-1)
                
                for j, evidence in enumerate(batch_evidences):
                    pred = torch.argmax(probs[j]).item()
                    self.verdicts.append({
                        "evidence": evidence,
                        "verdict": self.label_names[pred],
                        "scores": {name: float(probs[j][k]) for k, name in enumerate(self.label_names)}
                    })
            
            top_verdict_info = self.verdicts[0]
            top_verdict_label = top_verdict_info["verdict"]
            top_verdict_scores = top_verdict_info["scores"]
            
            if top_verdict_label == "entailment" and top_verdict_scores["entailment"] > 0.85:
                result = "TRUE"
            elif top_verdict_label == "contradiction" and top_verdict_scores["contradiction"] > 0.8:
                result = "FALSE"
            else:
                for v in self.verdicts[1:]:
                    if v["verdict"] == "contradiction" and v["scores"]["contradiction"] > 0.9:
                        result = "FALSE"
                        break
                else:
                    result = "NEUTRAL"
            
            return result, self.verdicts
        except Exception as e:
            raise RuntimeError(f"Classification failed | {e}")

    def __call__(self, claim, evidences):
        return self.classify(claim, evidences)

class summarizer:
    def __init__(self):
        self.model_name = "google/flan-t5-large"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Summarizer device:", self.device)
        
        try:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name, 
                dtype=dtype,
                low_cpu_mem_usage=True
            )
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(" Summarizer too big for GPU → using CPU")
                self.device = torch.device("cpu")
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            else:
                raise RuntimeError(f"Could not fetch model from Hugging Face | {e}")

    def forward(self, claim, top_evidence, verdict, max_input_len=2048, max_output_len=150):
        evidences = [e[1] for e in top_evidence]
        print("simple summazrizer is being used")

        if not evidences:
            raise ValueError("No evidence provided")
        
        evidence_text = "\n\n---\n\n".join(top_evidence[:3])
        
        input_text = f"""
        You are a fact-checking assistant.
        Claim: "{claim}"

        Evidence from reliable sources:
        {evidence_text}

        Classifier Verdict: {verdict}

        Task: 
        1. Confirm or revise the classifier verdict ({verdict}) using ONLY the evidence above.
        2. Explain briefly in 5 to 6 sentences, citing the evidence (no hallucinations).
        3. Keep everything strictly related to the claim.
        """

    
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_input_len
        ).to(self.device) 
        
        summary_ids = self.model.generate(
            inputs["input_ids"], 
            max_length=max_output_len, 
            num_beams=4, 
            early_stopping=True
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return verdict, summary
    
    def __call__(self, claim, top_evidence, verdict, max_input_len=1024, max_output_len=150):
        return self.forward(claim, top_evidence, verdict, max_input_len, max_output_len)
    

class FactChecker:
    def __init__(self):
        self.factcheck_api = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        self.google_search = "https://www.google.com/search"
        # self.reranker = None
        # self.classifier = None
        # self.summarizer = None

    def check_google_factcheck(self, claim: str , pages:int=5):
    
        load_dotenv()
        api_key = os.getenv("GOOGLE_FACT_CHECK_API")
        
        if not api_key:
            print("Google FactCheck API key not found")
            return None
        
        params = {
            'key': api_key,
            'query': claim,
            'languageCode': 'en-US',
            'pageSize': pages
        }
        
        try:
            response = requests.get(self.factcheck_api, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'claims' in data and data['claims']:# Returning the most relevant fact-check

                claim_data = data['claims'][0]
                review = claim_data.get('claimReview', [{}])[0]
                
                re= {
                    'claim': claim_data.get('text', ''),
                    'verdict': review.get('textualRating', 'Unknown'),
                    'summary': f"Rated {review.get('textualRating', 'Unknown')} by {review.get('publisher', {}).get('name', 'Unknown')}",
                    'source': review.get('url', ''),
                    # 'confidence': 'high',  # From official fact-checkers
                    'method': 'google_factcheck',
                    'URLs':"No url for this claim as it comes from Google fact check api"
                }
                return re
            
        except Exception as e:
            print(f"FactCheck API error: {e}")
        
        return {}

    def fetch_news_org(self, claim : str ,reranker, classifier, summarizer, page_size: int = 100, num_iter: int = 12, sort_by_index: int = 0):
        url_news_api = "https://newsapi.org/v2/everything"

        load_dotenv()
        api_key_news = os.getenv("NEWS_API")
        if not api_key_news:
            raise ValueError("NEWS_API environment variable not set")

        sort_by = ["relevancy", "popularity", "publishedAt"]
        if sort_by_index < 0 or sort_by_index >= len(sort_by):
            sort_by_index = 0

        news_ds = []

        for page in range(1, num_iter + 1):
            params = {
                "apiKey": api_key_news,
                "q": claim,
                "sortBy": sort_by[sort_by_index], 
                "pageSize": min(page_size, 100),
                "page": page,
            }

            try:
                res = requests.get(url_news_api, params=params, timeout=30)
                res.raise_for_status()

                if res.json().get("status") == "ok":
                    print(f"status | {res.json()['status']}")

                data = res.json().get("articles")
                if not data:
                    print(f"Could not find any article at page {page}")
                    break

                for article in data:
                    if not article.get("content"):
                        continue

                    news_ds.append({
                        "title": article.get("title", ""),
                        "text": (article.get("content", "") or "") + (article.get("description", "") or ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", {}).get("name", "No source available"),
                        "Published_Date": article.get("publishedAt", ""),
                    })

                if len(news_ds) >= res.json().get("totalResults", 0):
                    print(f"No More Results | reached {res.json().get('totalResults', 0)} Results")
                    break

            except requests.exceptions.RequestException as e:
                print(f"Request error on page(iteration) {page}: {e}")
                break
            except Exception as e:
                print(f"Unexpected error on page(iter) {page}: {e}")
                break

        print(f"Fetched {len(news_ds)} news articles")

        try:

            top_evidences = [dict.get('text') for dict in news_ds]
            urls = [dict.get('url') for dict in news_ds]

            reranked_articles = reranker.rerank_evidendce(claim,top_evidences)
            
            #  Classifying
            verdict,_ = classifier(claim,reranked_articles)
            
            #  Generating summary 
            verdict,summary = summarizer(claim,top_evidences,verdict)
            
            return {
                'claim': claim,
                'verdict': verdict,
                'summary': summary,
                'source': [arc.get('source','') for arc in news_ds],
                'method': 'NewsOrg API',
                'URLs':urls
            }
        
        except Exception as e:
            print("newsorg API failed", e)
            return {}
            
    
    def search_and_analyze_claim(self, claim: str ,reranker, classifier, summarizer):
        """
        Fallback method: Search web and analyze results with your models
        """
        print("No FactCheck result found, performing web analysis...")
        
        # self.classifier = Classifier()
        # self.summarizer = summarizer()
        # self.reranker = reranker()


        top_evidences,urls,article_list = self.google_news_search(claim)
        
        if not top_evidences:
            return {
                'claim': claim,
                'verdict': 'Unverifiable',
                'summary': 'No relevant sources found to verify this claim',
                # 'confidence': 'low',
                'method': 'web_search',
                'soruce':"No Source",
                'URLs':""
            }
        
        reranked_articles = reranker.rerank_evidendce(claim,top_evidences)
        
        verdict,_ = classifier(claim,reranked_articles)
        
        verdict,summary = summarizer(claim,top_evidences,verdict)
        
        return {
            'claim': claim,
            'verdict': verdict,
            'summary': summary,
            'source': [arc.get('source','') for arc in article_list],
            'method': 'web_analysis',
            'URLs':urls
        }
    
    def google_news_search(self,query:str,num_pages:int = 1):
        print("Searching the Web")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/115.0.0.0 Safari/537.36"
                        }
        
        articles_gg= []
        for pages in range(num_pages):
            params = {
                "q":query,
                "tbm":"nws",
                'start':int(pages) * 10 
                }

            try:
                res = requests.get(self.google_search,params=params,headers=headers,timeout=15)
                print(res)
                soup = BeautifulSoup(res.text,'html.parser')

                article_list=soup.select("div.SoaBEf a")
                if not article_list:
                    print("None Articles found")
                for article in article_list:
                    h1_tag = article.find('div', class_="n0jPhd ynAwRc MBeuO nDgy9d")
                    h1 = h1_tag.get_text(strip=True) if h1_tag else ""

                    h2_tag = article.find('div',class_ = "GI74Re nDgy9d")
                    h2 = h2_tag.get_text(strip=True) if h2_tag else ""
                    title = h1 + h2

                    a_url = article['href']
                    time_tag = article.find('div',class_="OSrXXb rbYSKb LfVVr")
                    time = time_tag.text if time_tag else "No time found"
                    source_tag = article.find('div',class_ = "MgUUmf NUnG9d")
                    source = source_tag.text if source_tag else "No source found"
                    try:
                        down = tra.fetch_url(a_url)
                        content = tra.extract(down) if down else "none extracted"
                        content = content if content else "No content extracted"
                    except Exception as e:
                        content = f"Error: {e}"

                    articles_gg.append({
                        "title":title,
                        'url':a_url,
                        'text':content,
                        'pblished_date':time,
                        'source':source
                    })
                
            except requests.exceptions.RequestException as e:
                print(f"Error Fething Google search | {e}")

            except Exception as e:
                print(f"Unforseen Error | {e}")

        top_evidences = [dict.get('text') for dict in articles_gg]
        urls = [dict.get('url') for dict in articles_gg]
        print("Web search Successfull")
        return top_evidences,urls,articles_gg
    

    
    def check_claim(self, claim: str, reranker, classifier, summarizer):
        """
        Main function to check a claim using the complete pipeline
        """
        print(f"Checking claim: '{claim}'")
        
        # First trying Google FactCheck API
        factcheck_result = self.check_google_factcheck(claim)
        
        if factcheck_result:
            print("Found result in FactCheck database")
            # return factcheck_result
        newsorg_result =self.fetch_news_org(claim, reranker, classifier, summarizer)
        if newsorg_result:
            print("Found result in NewsOrg database")
        
        # Fallback to web search 
        # print("No FactCheck result, falling back to web analysis")
        re =  self.search_and_analyze_claim(claim, reranker, classifier, summarizer)
        return re,factcheck_result,newsorg_result

class img_manipulation:
    def __init__(self):
        self.GEN_AI_IMAGE = pipeline("image-classification", model="umm-maybe/AI-image-detector", device=DEVICE)

    def Gen_AI_IMG(self, img_pth):
        try:
            with Image.open(img_pth) as img:
                img = img.convert('RGB')
                result = self.GEN_AI_IMAGE(img)
            proba = next((item['score'] for item in result if item['label'] == 'artificial'), 0.0)
            return proba * 100
        except Exception as e:
            print(f'AI image detection error: {e}')
            return 0.0

    def generated_image(self, img_pth, quality=90, scale=15):
        try:
            with Image.open(img_pth) as orig_img:
                orig_img = orig_img.convert('RGB')
                temp_path = 'temp_resaved.jpg'
                orig_img.save(temp_path, 'JPEG', quality=quality)
                with Image.open(temp_path) as resaved_img:
                    ela_image = ImageChops.difference(orig_img, resaved_img)
            os.remove(temp_path)
            ela_data = np.array(ela_image)
            mean_intensity = ela_data.mean()
            scaled_score = min(100, (mean_intensity / 25.0) * 100)
            
            ela_path = "ela_result.png"
            enhancer = ImageEnhance.Brightness(ela_image)
            max_diff = max(1, max([ex[1] for ex in ela_image.getextrema()]))
            ela_image_enhanced = enhancer.enhance(scale / max_diff)
            ela_image_enhanced.save(ela_path)
            return scaled_score, ela_path
        except Exception as e:
            print(f'ELA generation error: {e}')
            return 0.0, None

    def run_image_forensics(self, image_path):
        ai_score = self.Gen_AI_IMG(image_path)
        classic_score, ela_path = self.generated_image(image_path)
        return {
            "ai_generated_score_percent": ai_score,
            "classic_edit_score_percent": classic_score,
            "ela_image_path": ela_path
        }

class OCR:
    def __init__(self, key_path='GOOGLE_VISION_API.json'):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path
        self.client = vision.ImageAnnotatorClient()

    def _get_full_vision_analysis(self, img_pth):
        try:
            with open(img_pth, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            features = [{'type_': vision.Feature.Type.DOCUMENT_TEXT_DETECTION}, {'type_': vision.Feature.Type.SAFE_SEARCH_DETECTION}, {'type_': vision.Feature.Type.LANDMARK_DETECTION}, {'type_': vision.Feature.Type.LOGO_DETECTION}, {'type_': vision.Feature.Type.WEB_DETECTION}]
            response = self.client.annotate_image({'image': image, 'features': features})
            return response, None
        except Exception as e:
            return None, str(e)

    def get_in_image_anal(self, img_pth):
        response, error = self._get_full_vision_analysis(img_pth)
        if error: return {'error': error}
        report = {}
        if response.full_text_annotation: report['Extracted Text'] = response.full_text_annotation.text
        if response.safe_search_annotation:
            safe = response.safe_search_annotation
            report['Safe Search'] = {'adult': vision.Likelihood(safe.adult).name, 'violence': vision.Likelihood(safe.violence).name}
        entities = []
        if response.landmark_annotations: entities.extend([f'Landmark: {l.description}' for l in response.landmark_annotations])
        if response.logo_annotations: entities.extend([f'Logo: {l.description}' for l in response.logo_annotations])
        if entities: report['Identified Entities'] = entities
        return report

    def rev_img_search(self, img_pth):
        response, error = self._get_full_vision_analysis(img_pth)
        if error: return {'error': error}
        report = {}
        if response.web_detection and response.web_detection.pages_with_matching_images:
            matches = [{'title': p.page_title, 'url': p.url} for p in response.web_detection.pages_with_matching_images[:5]]
            report['Reverse Image Matches'] = matches
        return report

