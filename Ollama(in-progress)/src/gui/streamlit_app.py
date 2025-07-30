import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crewai_vision_tool.crew import CrewaiVisionTool
class MermaidGenerator:

    def generate_diagram(self, path):
        inputs = {
            'image_path': path
        }
        return CrewaiVisionTool().crew().kickoff(inputs=inputs)
    
    def mermaid_generation(self):
        if st.session_state.generating:
            st.session_state.final_diagram = self.generate_diagram(st.session_state.path)

        if st.session_state.final_diagram and st.session_state.final_diagram != "":
            with st.container():
                st.write("Mermaid Diagram successfully generated!")
                st.download_button(
                    label = "Download the mermaid diagram",
                    data = st.session_state.final_diagram,
                    file_name = "mermaid.mmd",
                    mime="text/plain"
                )
            st.session_state.generating = False

    def sidebar(self):
        with st.sidebar:
            st.title("Mermaid Diagram Generator")
            st.write("To generate the mermaid diagram, just copy-paste the path to the image you want to analyze in the box below and click on the button.")

            st.text_area("Image Path", key="path", placeholder="Enter the image path for the diagram")
            if st.button("Generate Mermaid Diagram"):
                st.session_state.generating = True

    def render(self):
        st.set_page_config(page_title="Mermaid Diagram Generator", page_icon="🧜‍♀️")

        if "path" not in st.session_state:
            st.session_state.path = ""

        if "final_diagram" not in st.session_state:
            st.session_state.final_diagram = ""

        if "generating" not in st.session_state:
            st.session_state.generating = False

        self.sidebar()
        self.mermaid_generation()

if __name__ == "__main__":
    MermaidGenerator().render()