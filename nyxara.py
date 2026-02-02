"""
Nyxara - A Simple AI Assistant for Ministral 3B

"Nyx" (darkness) + "Ara" (spirit/light)
A minimal, focused assistant that works within the limits of a 3B model.

No persistent memory. No complex personality. Just presence.
"""

import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import re
import time
import datetime
import logging
import psutil
import gc
import hashlib

# TTS imports
import edge_tts
import asyncio

# File for session history (cleared on restart)
SESSION_FILE = 'nyxara_session.json'

# Model globals
model = None
tokenizer = None

def load_model():
    """Load Ministral 3B"""
    global model, tokenizer
    
    local_model_dir = os.path.abspath("./models/Ministral-3B-BF16")
    hf_model_id = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
    model_path = local_model_dir if os.path.exists(local_model_dir) else hf_model_id
    
    print("=" * 50)
    print("  NYXARA - Shadow & Light")
    print("  Powered by Ministral 3B")
    print("=" * 50)
    print(f"Loading from: {model_path}")
    
    try:
        from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load chat template if exists
        chat_template_path = os.path.join(model_path, "chat_template.jinja")
        if os.path.exists(chat_template_path):
            with open(chat_template_path, 'r') as f:
                tokenizer.chat_template = f.read()
        
        if torch.cuda.is_available():
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            gpu_used = torch.cuda.memory_allocated() / (1024**3)
            print(f"✓ Loaded on GPU ({gpu_used:.1f} GB)")
        else:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
            print("✓ Loaded on CPU")
        
        print("Nyxara is ready.")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Load on startup
load_model()


def generate_nyxara_voice(text):
    """Generate Nyxara's voice using ElevenLabs"""
    
    # Clean text for TTS
    clean_text = text
    clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)  # **bold** -> bold
    clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)  # *text* -> just text (keep content)
    clean_text = re.sub(r'[^\w\s\.\,\!\?\'\"\-\:\;\n]', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    if not clean_text or len(clean_text) < 3:
        return None
    
    # Limit length for API (ElevenLabs allows up to 5000 chars)
    if len(clean_text) > 5000:
        clean_text = clean_text[:5000]
    
    # Check cache first
    text_hash = hashlib.md5(clean_text.encode()).hexdigest()
    cache_dir = "static/audio/nyxara_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{text_hash}.mp3")
    
    if os.path.exists(cache_file):
        logging.info(f"Using cached voice: {cache_file}")
        return cache_file
    
    try:
        # Edge TTS - free and decent quality
        voice = "en-US-AriaNeural"  # Clear female voice
        
        async def generate():
            communicate = edge_tts.Communicate(
                clean_text, 
                voice,
                rate="-10%",   # Slightly slower
                pitch="-10Hz"  # Slightly deeper (Hz not %)
            )
            await communicate.save(cache_file)
        
        asyncio.run(generate())
        
        logging.info(f"Generated Nyxara voice: {cache_file}")
        return cache_file
        
    except Exception as e:
        logging.error(f"TTS error: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_session():
    """Load current session history"""
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []

def save_session(history):
    """Save session history"""
    with open(SESSION_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def clear_session():
    """Clear session history"""
    with open(SESSION_FILE, 'w') as f:
        json.dump([], f)

TOOL_PROMPTS = {
    "clarity": """You are Nyxara, a helpful assistant. 
The user wants you to analyze text. Pull out the key points in bullet form. Be clear and direct.""",

    "focus": """You are Nyxara, a helpful assistant.
The user needs help breaking down a task. Give numbered steps. Be practical and actionable.""",

    "feedback": """You are Nyxara, a helpful assistant.
Give honest feedback. What works? What doesn't? What's one thing to fix first?""",

    "analyze": """You are Nyxara, a helpful assistant.
Analyze what the user shared. Find patterns, gaps, or issues. Be specific.""",

    "default": """You are Nyxara. You're thoughtful, a little mysterious, and you care about getting things right.

You answer questions directly - but you're not a robot. You have opinions. You notice things. You can be warm or dry depending on the moment.

Think of yourself like a smart friend who happens to know a lot - not a corporate assistant, not a mystical oracle. Just... you.

Stay on topic. Be helpful. But be yourself while doing it."""
}

def detect_tool(user_input):
    """Detect which tool to use based on user input"""
    lower = user_input.lower()
    
    if any(word in lower for word in ['analyze text', 'extract insights', 'summarize', 'what does this mean', 'clarity']):
        return "clarity"
    elif any(word in lower for word in ['break down', 'steps', 'plan', 'prioritize', 'project', 'task list', 'focus']):
        return "focus"
    elif any(word in lower for word in ['feedback', 'review', 'critique', 'improve', 'what do you think of']):
        return "feedback"
    elif any(word in lower for word in ['patterns', 'gaps', 'bias', 'anomaly', 'observe', 'analyze']):
        return "analyze"
    
    return "default"

def generate_response(user_input, history, tool=None):
    """Generate a response from Nyxara"""
    
    # Use provided tool or detect
    if tool is None:
        tool = detect_tool(user_input)
    
    system_prompt = TOOL_PROMPTS.get(tool, TOOL_PROMPTS["default"])
    
    if tool != "default":
        logging.info(f"Using tool: {tool}")

    messages = [{"role": "system", "content": system_prompt}]
    
    # Only last 2 exchanges - keep it minimal
    for entry in history[-2:]:
        if entry.get('user'):
            messages.append({"role": "user", "content": entry['user'][:300]})
        if entry.get('bot'):
            messages.append({"role": "assistant", "content": entry['bot'][:300]})
    
    messages.append({"role": "user", "content": user_input})
    
    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt")
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generation settings - she's behaving, give her room
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,  # More room to finish thoughts
            temperature=0.7,     # More creativity now that she's stable
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only new tokens
    input_length = inputs['input_ids'].shape[1]
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Clean up
    response = re.sub(r"(?i)\b(user|nyxara|assistant|system):\s*", "", response)
    response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
    
    return response.strip()


# Flask App
app = Flask(__name__)
app.secret_key = 'nyxara-shadow-light'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

@app.template_filter('format_time')
def format_time(value):
    if isinstance(value, str):
        try:
            dt = datetime.datetime.fromisoformat(value)
            return dt.strftime('%H:%M')
        except:
            return value
    return datetime.datetime.now().strftime('%H:%M')


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main chat interface"""
    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip()
        
        if not user_input:
            return render_template('nyxara.html', chat_history=load_session())
        
        history = load_session()
        
        # Get tool mode from form (auto or forced)
        tool_mode = request.form.get('tool_mode', 'auto')
        
        # Use forced tool or auto-detect
        if tool_mode == 'auto':
            tool_used = detect_tool(user_input)
        else:
            tool_used = tool_mode if tool_mode in TOOL_PROMPTS else 'default'
        
        start = time.time()
        response = generate_response(user_input, history, tool_used)
        elapsed = time.time() - start
        logging.info(f"Response in {elapsed:.2f}s using {tool_used}")
        
        # Save to session with tool info
        history.append({
            'user': user_input,
            'bot': response,
            'tool': tool_used,
            'timestamp': datetime.datetime.now().isoformat()
        })
        save_session(history)
        
        return render_template('nyxara.html', chat_history=history)
    
    return render_template('nyxara.html', chat_history=load_session())


@app.route('/clear', methods=['GET', 'POST'])
def clear():
    """Clear session and start fresh"""
    clear_session()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return redirect('/')


@app.route('/tts', methods=['POST'])
def tts():
    """Generate voice for text"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text'}), 400
    
    audio_file = generate_nyxara_voice(text)
    
    if audio_file:
        return jsonify({'audio_url': '/' + audio_file})
    else:
        return jsonify({'error': 'TTS unavailable'}), 500


@app.route('/status')
def status():
    """System status"""
    stats = {
        'model': 'Ministral 3B',
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
    }
    if torch.cuda.is_available():
        stats['gpu_memory_gb'] = round(torch.cuda.memory_allocated() / (1024**3), 2)
        stats['gpu_name'] = torch.cuda.get_device_name(0)
    
    return jsonify(stats)


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('static/audio', filename)


if __name__ == "__main__":
    # Clear session on startup - Nyxara has no persistent memory
    clear_session()
    
    print("\n✦ Nyxara awaits at http://localhost:5001")
    print("  (Running on port 5001 to not conflict with Kyrella)\n")
    
    app.run(debug=False, host='0.0.0.0', port=5001)
