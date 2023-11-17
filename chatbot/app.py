import streamlit as st
from rag_function import rag_func

st.set_page_config(
    page_title="MedBot",
    page_icon="👨‍⚕️",
)
header = {
    "authorization": st.secrets["OPENAI_API_KEY"],
    "content-type": "application/json"
}

st.markdown("<h1 style='text-align: center;'>MedBot</h1>", unsafe_allow_html=True)
st.header("", divider = 'rainbow')
# set  initial message
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello there, how can I help you today?"}
    ]

# display messages
if "messages" in st.session_state.keys():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt })
    with st.chat_message("user"):
        st.write(user_prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading....."):
            ai_response = rag_func(user_prompt)
            st.write(ai_response)
    new_ai_messages =  {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_messages)