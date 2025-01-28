import os
import tempfile
import numpy as np
import trimesh
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

# Set an environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Set up the Streamlit app
st.set_page_config(page_title="LLaMA-Mesh", layout="wide")
st.title("LLaMA-Mesh")
st.markdown('''<div>
    <h1 style="text-align: center;">LLaMA-Mesh</h1>
    <div>
    <a style="display:inline-block" href="https://research.nvidia.com/labs/toronto-ai/LLaMA-Mesh/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
    <a style="display:inline-block; margin-left: .5em" href="https://github.com/nv-tlabs/LLaMA-Mesh"><img src='https://img.shields.io/github/stars/nv-tlabs/LLaMA-Mesh?style=social'/></a>
    </div>
    <p>LLaMA-Mesh: Unifying 3D Mesh Generation with Language Models.<a style="display:inline-block" href="https://research.nvidia.com/labs/toronto-ai/LLaMA-Mesh/">[Project Page]</a> <a style="display:inline-block" href="https://github.com/nv-tlabs/LLaMA-Mesh">[Code]</a></p>
    <p> Notice: (1) This demo supports up to 4096 tokens due to computational limits, while our full model supports 8k tokens. This limitation may result in incomplete generated meshes. To experience the full 8k token context, please run our model locally.</p>
    <p>(2) We only support generating a single mesh per dialog round. To generate another mesh, click the "clear" button and start a new dialog.</p>
    <p>(3) If the LLM refuses to generate a 3D mesh, try adding more explicit instructions to the prompt, such as "create a 3D model of a table <strong>in OBJ format</strong>." A more effective approach is to request the mesh generation at the start of the dialog.</p>
</div>''', unsafe_allow_html=True)

# Load the tokenizer and model
model_path = "Zhengyi/LLaMA-Mesh"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def apply_gradient_color(mesh_text):
    """
    Apply a gradient color to the mesh vertices based on the Y-axis and save as GLB.
    Args:
        mesh_text (str): The input mesh in OBJ format as a string.
    Returns:
        str: Path to the GLB file with gradient colors applied.
    """
    # Load the mesh
    temp_file = tempfile.NamedTemporaryFile(suffix=f"", delete=False).name
    with open(temp_file+".obj", "w") as f:
        f.write(mesh_text)
    mesh = trimesh.load_mesh(temp_file+".obj", file_type='obj')

    # Get vertex coordinates
    vertices = mesh.vertices
    y_values = vertices[:, 1]  # Y-axis values

    # Normalize Y values to range [0, 1] for color mapping
    y_normalized = (y_values - y_values.min()) / (y_values.max() - y_values.min())

    # Generate colors: Map normalized Y values to RGB gradient (e.g., blue to red)
    colors = np.zeros((len(vertices), 4))  # RGBA
    colors[:, 0] = y_normalized  # Red channel
    colors[:, 2] = 1 - y_normalized  # Blue channel
    colors[:, 3] = 1.0  # Alpha channel (fully opaque)

    # Attach colors to mesh vertices
    mesh.visual.vertex_colors = colors

    # Export to GLB format
    glb_path = temp_file+".glb"
    with open(glb_path, "wb") as f:
        f.write(trimesh.exchange.gltf.export_glb(mesh))
    
    return glb_path

def chat_llama3_8b(message: str, history: list, temperature: float, max_new_tokens: int) -> str:
    """
    Generate a streaming response using the llama3-8b model.
    Args:
        message (str): The input message.
        history (list): The conversation history.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The generated response.
    """
    conversation = []
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    max_new_tokens=4096
    temperature=0.9
    generate_kwargs = dict(
        input_ids= input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=terminators,
    )
    if temperature == 0:
        generate_kwargs['do_sample'] = False
        
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)

# Streamlit UI
st.markdown("### Chat with LLaMA-Mesh Model")
user_input = st.text_input("Enter your message:")
temperature = st.slider("Temperature", 0.0, 1.0, 0.9, 0.1)
max_new_tokens = st.slider("Max new tokens", 128, 4096, 4096)

if st.button("Generate Response"):
    history = []
    response = chat_llama3_8b(user_input, history, temperature, max_new_tokens)
    st.text_area("Response", response)

st.markdown("### 3D Mesh Visualization")
mesh_input = st.text_area("3D Mesh Input (OBJ format)")
if st.button("Visualize 3D Mesh"):
    glb_path = apply_gradient_color(mesh_input)
    st.write("3D Mesh with Gradient Color")
    st.download_button("Download GLB", data=open(glb_path, "rb"), file_name="gradient_mesh.glb")
