# test app 1
#  import streamlit as st

# st.title("Counter Example")

# print(111)
# if "count" not in st.session_state:
#     print(222)
#     st.session_state.count = 0

# print(333)
# if 'count' not in st.session_state:
#    print(444)
#    st.session_state['count'] = 0



# increment = st.button("Increment")
# if increment:
#     st.session_state.count += 1

# st.write('Count = ', st.session_state.count)


# test app 2
# ---- Modules ------- 

import streamlit as st
import pandas as pd
import plotly.express as px

# --- Initialising SessionState ---
if "load_state" not in st.session_state:
    st.session_state.load_state = False


st.header("Fruits List")
# ---- Creating Dictionary ----
_dic = { 'Name': ['Mango', 'Apple', 'Banana'],
         'Quantity': [45, 38, 90]}

print('start')
load = st.button('Load Data')
print(load)
if load:
    print(111)
    st.session_state.load_state = True
    _df = pd.DataFrame(_dic)
    st.write(_df)
   
   # ---- Plot types -------
    opt = st.radio('Plot type :',['Bar', 'Pie'])
    if opt == 'Bar':
        fig = px.bar(_df, x= 'Name',y = 'Quantity',title ='Bar Chart')
        st.plotly_chart(fig)
   
    else:     
        fig = px.pie(_df,names = 'Name',values = 'Quantity',title ='Pie Chart')
        st.plotly_chart(fig)