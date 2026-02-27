import streamlit as st
import streamlit_utils as st_utils
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="Post Questionnaire",
    page_icon="📝",
)

if not "participant_registered" in st.session_state or not st.session_state.get("participant_registered", False):
    switch_page("Introduction")

st_utils.display_sidebar()
st_utils.display_post_questionnaire()
