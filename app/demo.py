import sys
from pathlib import Path

import streamlit as st


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))


# --------------------------------------------------
# Load our AI support agent
# --------------------------------------------------

@st.cache_resource
def load_agent():
    import agent
    return agent


agent = load_agent()


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Spotify AI Support Agent",
    page_icon="🎧",
    layout="centered"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🎧 Spotify AI Support Agent")

st.write(
    "Ask a Spotify support question. "
    "The agent classifies the issue, retrieves a similar historical "
    "support case, drafts a grounded reply, and decides whether to "
    "auto-handle or escalate."
)

st.divider()


# --------------------------------------------------
# Customer message
# --------------------------------------------------

customer_message = st.text_area(
    "💬 Customer Message",
    placeholder="Example: I paid for Premium but my account still shows Free.",
    height=120
)


# --------------------------------------------------
# Run agent
# --------------------------------------------------

if st.button("🤖 Get Support Reply", use_container_width=True):

    if not customer_message.strip():

        st.warning("Please enter a customer message.")

    else:

        with st.spinner("Analyzing customer message..."):

            result = agent.run_agent(customer_message.strip())


        # --------------------------------------------------
        # Customer-facing reply
        # --------------------------------------------------

        st.subheader("💬 Bot Reply")

        st.info(result["reply"])


        # --------------------------------------------------
        # Internal agent analysis
        # --------------------------------------------------

        st.divider()

        st.subheader("🔍 Agent Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Intent",
                result["intent"]
            )

            st.metric(
                "Intent Confidence",
                f'{result["confidence"]:.2%}'
            )

        with col2:

            st.metric(
                "Historical Similarity",
                f'{result["similarity"]:.2%}'
            )

            st.metric(
                "Decision",
                result["decision"]
            )


        st.write("**Reason:**")
        st.write(result["reason"])


        st.write("**Historical Grounding Case:**")
        st.write(result["source_case"])