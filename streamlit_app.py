import streamlit as st



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