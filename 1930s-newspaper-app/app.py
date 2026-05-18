import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
import sys
import os
import re
import uuid
import datetime
import gc
import shutil
import requests

# Add local talkie module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "talkie", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "talkie"))
try:
    from talkie import Message as TalkieMessage, Talkie
    from generate_mlx import TalkieMLX
except ImportError as e:
    print(f"Talkie module not found or import error. Please ensure it is cloned correctly. Error: {e}")

# ==========================================
# 1. Authentic JCross V7 Spatial Memory
# ==========================================
class AuthenticJCrossMemory:
    """
    Authentic implementation derived from Verantyx Cortex V7.
    Uses pure text-based '.jcross' files with precise cognitive block formatting.
    """
    def __init__(self, base_dir="./verantyx_jcross_v7"):
        self.base_dir = base_dir
        self.zones = ["l1_topology", "front", "near", "mid", "deep"]
        self.boot_system()
        
    def boot_system(self):
        for zone in self.zones:
            os.makedirs(os.path.join(self.base_dir, zone), exist_ok=True)
            
        # Initialize authentic L1/L2 Dictionary nodes if empty
        l1_dir = os.path.join(self.base_dir, "l1_topology")
        if not os.path.exists(os.path.join(l1_dir, "L1_INDEX.jcross")):
            self.write_raw_node("l1_topology", "L1_INDEX", """■ JCROSS_L1_TOPOLOGY
【空間座相】 [Z:-1]
【位相タグ】 [標: 指示] [認: 1.0] [視: 0.8]
【メタデータ】 Initial topology mapping for Verantyx 1930s environment
""")
            
        front_dir = os.path.join(self.base_dir, "front")
        if not os.path.exists(os.path.join(front_dir, "DICT_1930.jcross")):
            self.write_raw_node("front", "DICT_1930", """■ JCROSS_NODE_DICT_1930
【空間座相】 [Z:0]
【次元概念】 #Design #1930 #Newspaper #Aesthetics
【操作軌道】 [引: CORE_PERSONA]
[本質記憶]
//! OP.SET_COLOR("Sepia Paper", "background-color: #f4ecd8; color: #2c241b;")
//! OP.USE_CSS_COLUMNS("Columns", "column-count: 2 or 3; column-gap: 20px; column-rule: 1px solid #3b2f2f;")
//! OP.USE_BORDERS("Dividers", "border-top: 4px double #1a1a1a; border-bottom: 2px solid #1a1a1a;")
//! [HTML/CSS Structural Dictionary] Use strictly <div> with the above inline styles. Avoid markdown.
""")

    def write_raw_node(self, zone, name, content):
        filepath = os.path.join(self.base_dir, zone, f"{name}.jcross")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def write_generated_node(self, zone, cognition_text):
        if not cognition_text: return
        node_id = str(uuid.uuid4())[:8]
        filepath = os.path.join(self.base_dir, zone, f"tm_{node_id}.jcross")
        
        # Replace the generic NODE_current with the new ID
        cognition_text = cognition_text.replace("JCROSS_NODE_current", f"JCROSS_NODE_{node_id}")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{cognition_text}\n")
        print(f"[System] Cortex Node Crystallized: {filepath}")

    def load_zone(self, zone):
        zone_dir = os.path.join(self.base_dir, zone)
        nodes = []
        for filename in os.listdir(zone_dir):
            if filename.endswith(".jcross"):
                with open(os.path.join(zone_dir, filename), "r", encoding="utf-8") as f:
                    nodes.append(f.read())
        return nodes
        
    def get_front_injection(self):
        active_nodes = self.load_zone("front")
        return "\n\n".join(active_nodes)
        
    def search_jcross(self, query):
        """
        Simulates the Rust query_jcross logic by scanning [本質記憶] blocks in 'near' zone.
        """
        near_nodes = self.load_zone("near")
        for node in near_nodes:
            # Extract essential memory block
            match = re.search(r"\[本質記憶\](.*)", node, re.DOTALL)
            if match:
                essence = match.group(1)
                if query.lower() in essence.lower():
                    # Extract the translation from the response format
                    trans_match = re.search(r"Translation:\s*(.*)", essence)
                    if trans_match:
                        return trans_match.group(1).strip()
        return None

    def migrate_memory(self):
        """
        Simulates Verantyx memory migration: near -> mid -> deep
        """
        near_dir = os.path.join(self.base_dir, "near")
        mid_dir = os.path.join(self.base_dir, "mid")
        deep_dir = os.path.join(self.base_dir, "deep")
        
        # Move older near nodes to mid
        near_files = sorted(os.listdir(near_dir))
        if len(near_files) > 5:
            for f in near_files[:-5]:
                shutil.move(os.path.join(near_dir, f), os.path.join(mid_dir, f))
                
        # Move older mid nodes to deep
        mid_files = sorted(os.listdir(mid_dir))
        if len(mid_files) > 20:
            for f in mid_files[:-20]:
                shutil.move(os.path.join(mid_dir, f), os.path.join(deep_dir, f))

memory_engine = AuthenticJCrossMemory()

# ==========================================
# 2. Historical Agent Loading
# ==========================================
HF_MODELS = {"historical": "talkie-1930-13b-base"}

historical_model = None
model_load_lock = threading.Lock()

def load_historical():
    global historical_model
    model_id = HF_MODELS["historical"]
    if historical_model is None:
        print(f"[Historical Agent] Loading MLX Native Ouroboros Architecture for {model_id}...")
        historical_model = TalkieMLX(model_id)

def unload_historical():
    global historical_model
    historical_model = None
    gc.collect()
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

# ==========================================
# 3. Verantyx V7 Loop (Agents)
# ==========================================

class HistoricalReporterAgent:
    @staticmethod
    def generate_article(news_text, max_tokens=500):
        # We extract a concise prefix for the event to ensure the model doesn't just repeat a massive modern text
        first_sentence = news_text.split('.')[0] + "." if '.' in news_text else news_text[:100]
        
        headline = "Special Report"
        subtitle = "Authorities Investigating Strange Occurrences"
        event = first_sentence
        
        # Time-Shift Prefix Autocomplete
        prefix = f"""THE NEW YORK TIMES - October 24, 1930

[HEADLINE]: {headline}
[SUBTITLE]: {subtitle}
[BYLINE]: By Arthur Conan, Senior Correspondent

NEW YORK — Yesterday, {event} """

        generator = historical_model.generate(
            prefix,
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        generated_text = ""
        for token_str in generator:
            generated_text += token_str
            paragraphs = generated_text.strip().split('\n\n')
            formatted_paragraphs = "".join([f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs if p.strip()])
            
            html = f"""<div style="background-color: #f4ecd8 !important; color: #2c241b !important; padding: 20px; font-family: serif;">
        <h1 style="text-align: center; font-size: 2.5em; border-bottom: 2px solid #1a1a1a; margin-bottom: 5px; color: #2c241b !important;">{headline}</h1>
        <h3 style="text-align: center; font-size: 1.5em; font-style: italic; margin-top: 0; padding-bottom: 15px; border-bottom: 4px double #1a1a1a; color: #2c241b !important;">{subtitle}</h3>
        <div style="column-count: 2; column-gap: 20px; column-rule: 1px solid #3b2f2f; margin-top: 20px; font-size: 1.1em; line-height: 1.6; color: #2c241b !important;">
            <p style="color: #2c241b !important;"><strong>NEW YORK — </strong>Yesterday, {event} {formatted_paragraphs}</p>
        </div>
    </div>"""
            yield html

class Orchestrator:
    def process(self, news_text):
        with model_load_lock:
            yield "Initializing Talkie 13B Engine...", "<div style='color: #888;'>Loading System Memory...</div>"
            display_str = "HEADLINE: Special Report\nEVENT: Generating directly via Talkie 13B Ouroboros..."
            yield display_str, "<div style='color: #888;'>System Memory Loaded. Direct Model Activation.</div>"
            
            load_historical()
            for html_chunk in HistoricalReporterAgent.generate_article(news_text, max_tokens=800):
                yield display_str, html_chunk
            unload_historical()
            
            # Migrate memory: near -> mid -> deep
            memory_engine.migrate_memory()

orchestrator = Orchestrator()

def ui_handler(news_text):
    if not news_text:
        yield "Please enter a news item.", "<div>Please enter a news item.</div>"
        return
    for result in orchestrator.process(news_text):
        yield result

# ==========================================
# 4. UI Setup
# ==========================================
with gr.Blocks(theme=gr.themes.Monochrome(), title="Authentic Verantyx V7 Proxy") as app:
    gr.Markdown("# 🧠 Talkie 13B Ouroboros Native Generation")
    gr.Markdown("Directly generating a 1930s-style newspaper article using the MLX-accelerated Talkie 13B base model.")
    
    with gr.Row():
        with gr.Column(scale=2):
            news_input = gr.Textbox(label="Current News (Input short prompt)", lines=5)
            generate_btn = gr.Button("⚙️ Generate 1930s Article", variant="primary")
            abstract_output = gr.Textbox(label="Status", interactive=False, lines=2)
        with gr.Column(scale=3):
            article_output = gr.HTML(label="1930s Newspaper Layout")
            
    generate_btn.click(fn=ui_handler, inputs=[news_input], outputs=[abstract_output, article_output])

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        # Initialize MPS backend on the main thread to prevent background thread deadlocks
        torch.zeros(1).to("mps")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
