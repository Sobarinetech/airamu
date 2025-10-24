import streamlit as st
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="AgentPI Chat", page_icon="🧠", layout="centered")

# --- HEADER ---
st.title("🧠 AgentPI Interactive Chat")
st.caption("Streamlit app connected to Supabase Edge Function: `agentpi-api`")

# --- LOAD CREDENTIALS FROM st.secrets ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_API_KEY = st.secrets["SUPABASE_API_KEY"]
USER_TOKEN = st.secrets["USER_TOKEN"]

# --- API ENDPOINT ---
url = f"{SUPABASE_URL}/functions/v1/agentpi-api"

# --- SESSION STATE ---
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# --- FUNCTION TO CALL AGENTPI ---
def send_to_agentpi(message: str):
    headers = {
        "Authorization": f"Bearer {USER_TOKEN}",
        "apikey": SUPABASE_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "message": message,
        "conversationHistory": st.session_state.conversation_history,
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        return {"error": str(e)}

    if result.get("success"):
        st.session_state.conversation_history = result["data"]["conversationHistory"]
        return {
            "response": result["data"]["response"],
            "apiExecutions": result["data"].get("apiExecutions", []),
        }
    else:
        return {"error": result.get("message", "Unknown error")}

# --- UI ---
st.write("Send a message to AgentPI below:")
user_input = st.text_area(
    "Your Message:", placeholder="Example: Parse this PDF: https://example.com/doc.pdf"
)

if st.button("Send"):
    if user_input.strip():
        with st.spinner("Processing your request..."):
            output = send_to_agentpi(user_input)
        if "error" in output:
            st.error(output["error"])
        else:
            response = output["response"]
            apis_used = output.get("apiExecutions", [])
            st.success("✅ Response received!")
            st.session_state.chat_log.append(("You", user_input))
            st.session_state.chat_log.append(("AgentPI", response))

            # --- FIXED SECTION: Convert API dicts to readable strings ---
            if apis_used:
                try:
                    apis_str_list = []
                    for api in apis_used:
                        if isinstance(api, dict):
                            # Try to show main identifier
                            apis_str_list.append(api.get("name") or api.get("type") or str(api))
                        else:
                            apis_str_list.append(str(api))
                    st.session_state.chat_log.append(("APIs Used", ", ".join(apis_str_list)))
                except Exception as e:
                    st.session_state.chat_log.append(("APIs Used", f"[Error parsing API data: {e}]"))
    else:
        st.warning("Please type a message before sending.")

# --- DISPLAY CONVERSATION HISTORY ---
st.subheader("💬 Conversation History")
for role, content in st.session_state.chat_log:
    if role == "You":
        st.markdown(f"**🧑 You:** {content}")
    elif role == "AgentPI":
        st.markdown(f"**🤖 AgentPI:** {content}")
    else:
        with st.expander(f"🔧 {role} details"):
            st.write(content)

# --- CLEAR CHAT ---
if st.button("Clear Chat"):
    st.session_state.conversation_history = []
    st.session_state.chat_log = []
    st.experimental_rerun()
