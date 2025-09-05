import streamlit as st
import os
from main import app, AgentState

st.title("LangGraph Research Pipeline")

# User input for research topic
topic = st.text_input("Enter the topic you want to research:")

if st.button("Run Research Pipeline") and topic:
    st.info(f"Running research pipeline for: {topic}")
    inputs = {"research_topic": topic}
    # Run the pipeline and collect outputs
    output_states = []
    for output in app.stream(inputs):
        output_states.append(output)
    final_state = app.invoke(inputs)
    st.success("Pipeline complete!")
    st.subheader("Final Report")
    st.markdown(final_state['report'])
    # Optionally, show intermediate outputs
    with st.expander("Show intermediate agent outputs"):
        for i, state in enumerate(output_states):
            st.write(f"Step {i+1}", state)
    # Download report
    st.download_button("Download Report", final_state['report'], file_name="research_report.md")
