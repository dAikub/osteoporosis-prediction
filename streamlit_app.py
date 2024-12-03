import streamlit as st


st.markdown("""
<style>
.st-emotion-cache-qnyxd6.ef3psqc5
{
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

pg1 = st.Page(
    page="views/main_page.py",
    title="Prediction",
    icon=":material/analytics:",
    default=True,

)


pg2 = st.Page(
    page="views/eval.py",
    title="Data",
    icon=":material/bar_chart_4_bars:",
)


pg = st.navigation(pages=[pg1,pg2])
pg.run()
