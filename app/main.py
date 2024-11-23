import streamlit as st

# set page configuration
st.set_page_config(
    page_title="Classifier",
    page_icon = ':bar_chart:',
    layout="wide",
    initial_sidebar_state="expanded",
)
st.image('images/deloitte_logo.png', width=100)
st.title('Classification Model Powered by SFL Scientific (A Deloitte Business)')
st.sidebar.image('images/sfl_logo.png', use_column_width=True)

menu_1= ['Option A', 'Option B', 'Option C', 'Option D']
choice_1 = st.sidebar.selectbox('Select Option', menu_1)

menu_2= ['Option 1', 'Option 2', 'Option 3', 'Option 4']
choice_1 = st.sidebar.selectbox('Select Option', menu_2)