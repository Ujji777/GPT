import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Strict GPU-only check
if not torch.cuda.is_available():
    st.error("🚫 No GPU detected. Please run this app on a GPU-enabled environment.")
    st.stop()

# Display GPU details
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
st.write(f"**Using GPU:** {gpu_name}")
st.write(f"**VRAM:** {vram_gb:.2f} GB")

# Model and quantization configuration
model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

st.write("📦 **Loading model and tokenizer...**")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="cuda"
)

# Optionally compile the model for speedup (PyTorch 2.0+)
try:
    model = torch.compile(model)
    st.write("✅ Model compiled with torch.compile()")
except Exception as e:
    st.write("ℹ️ torch.compile() not supported or encountered an issue:", e)

model.eval()

# Define the chat function
def chat(prompt: str) -> str:
    formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Disable gradients to speed up inference
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Adjust if you need shorter or longer responses
            temperature=0.5,     # Lower temperature for deterministic results
            top_p=0.9,
            do_sample=False,     # Greedy decoding is faster
            use_cache=True
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    try:
        response = decoded.split("<|assistant|>\n")[1].split("<|user|>")[0].strip()
    except Exception as e:
        response = decoded.split("<|assistant|>\n")[1].strip()
    return response

# Streamlit interface
st.title("GPT-Style Chatbot Interface")

st.markdown(
    """
    Enter your query below and the model will generate a response.
    """
)

# Input from the user
user_input = st.text_input("Enter your query:")

if st.button("Submit") and user_input:
    with st.spinner("Generating response..."):
        response = chat(user_input)
    st.markdown("**Response:**")
    st.write(response)
