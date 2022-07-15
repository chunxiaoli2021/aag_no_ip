import pandas as pd
import numpy as np
import streamlit as st
from functions import load_input_file, preprocessing, run_NO_model, postprocessing, save_output_to_excel

@st.cache(suppress_st_warning=True)
def fup(file_object):  
    print('run fup()')
    if file_object is not None:
        with st.spinner('Wait for files uploading...'):
            dfs = load_input_file(uploaded_file)
        st.info("Files successfully uploaded!")   
    return dfs

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def prep(dfs):
    print('run prep()')
    with st.spinner('Wait for preprocessing...'):
        inputs = preprocessing(dfs)
    st.success('Preprocessing complete!')
    return inputs

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def run_model(inputs, scenario):
    print('run run_model()')
    with st.spinner('Wait for network optimization model running...'):
        costs, outputs = run_NO_model(inputs, scenario)
    st.success('Network optimization complete!')
    return costs, outputs

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def post(dfs, inputs, outputs):
    print('run post()')
    with st.spinner('Wait for postprocessing...'):
        res_dfs = postprocessing(dfs, inputs, outputs)
    st.success('Postprocessing complete!')
    return res_dfs

st.title("Network Optimization")
        
st.markdown("### Upload model input file")
uploaded_file = st.file_uploader("Upload .XLSX file", type=".xlsx")

use_example_file = st.checkbox(
    "Use example file", False, help="Use in-built example file to demo the app"
)

if use_example_file:
    uploaded_file = "model_input_20220622_EN.xlsx"

print('starting....')
if uploaded_file:
    print(111)
    dfs = fup(uploaded_file)

    st.markdown("### Data preview")
    st.dataframe(pd.DataFrame({'Tables': dfs.keys()}))

    st.markdown("### Preprocessing")
    print(222)
    inputs = prep(dfs)
    print(333)
    st.markdown("### Set up Scenario parameters")
    with st.form(key="my_form"):
        scenario = {}
        scenario['dc_handling_cap'] = st.checkbox('1. DC Handling Capacity Constraint')
        #st.select_slider("DC Handling Capacity Constraint:", ["On", "Off"])
        scenario['dc_storage_cap'] = st.checkbox('2. DC Storage Capacity Constraint', value = True)
        scenario['factory_handling_cap'] = st.checkbox('3. Factory Handling Capacity Constraint')
        scenario['factory_storage_cap'] = st.checkbox('4. Factory Storage Capacity Constraint', value = True)
        scenario['perc_demand_satisfied'] = st.checkbox('5. Percentage of Demand Satisfied Constraint')
        scenario['dc_initial_inv'] = st.checkbox('6. DC Initial Inventory Constraint', value = True)
        scenario['factory_initial_inv'] = st.checkbox('7. Factory Initial Inventory Constraint', value = True)
        scenario['fix_dcs_open'] = st.checkbox('8. Fixed Number of DCs Open Constraint', value = True)
        scenario['fix_factories_open'] = st.checkbox('9. Fixed Number of Factories Open Constraint', value = True)
        scenario['num_dcs_open'] = st.number_input('10. Number of Open DCs', min_value = 1, max_value = len(inputs['dcs']), step = 1, value = len(inputs['dcs']))
        scenario['num_factories_open'] = st.number_input('11. Number of Open Factories', min_value = 1, max_value = len(inputs['factories']), step = 1, value = len(inputs['factories']))
        scenario['hist_fc'] = st.checkbox('12. Historical Customer Flows Constraint')
        scenario['hist_ff'] = st.checkbox('13. Historical Intersite Flows Constraint')
        scenario['hist_prod'] = st.checkbox('14. Historical Production Flows Constraint')
        scenario['hist_no_prod'] = st.checkbox('15. Historical Non-Production Flows Constraint')

        submitted = st.form_submit_button(label="Submit")
    
    if submitted:
        st.markdown("### Run Network Optimization model")

        if not scenario:
            print(111111)
            st.warning("Please select and submit scenario parameters.")
            st.stop()
        
        costs, outputs = run_model(inputs, scenario)

        st.markdown("### Postprocessing")
        res_dfs = post(dfs, inputs, outputs)

        st.markdown("### Output preview")
        st.dataframe(pd.DataFrame({'Summary Tables': [x for x in res_dfs.keys() if 'summary' in x]}))
    #     
    # with st.spinner('Wait for solving network optimization model...'):
    #     costs, outputs = run_NO_model(inputs, scenario)
    # st.success('Done!')       