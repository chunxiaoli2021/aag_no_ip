import pandas as pd
import numpy as np
import streamlit as st
from functions import load_input_file, preprocessing, run_NO_model, postprocessing, save_output_to_excel
import time

# --- Initialising SessionState ---
if 'load_state' not in st.session_state:
    st.session_state.load_state = True
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'dfs' not in st.session_state:
    st.session_state.dfs = None
if 'inputs' not in st.session_state:
    st.session_state.inputs = None


# --- Streamlit App ---
st.title("Network Optimization")
        
st.markdown("### Upload model input file")
st.session_state.uploaded_file = st.file_uploader("Upload .XLSX file", type=".xlsx")

use_example_file = st.checkbox(
    "Use example file", False, help="Use in-built example file to demo the app"
)

if use_example_file:
    st.session_state.uploaded_file = "model_input_20220622_EN.xlsx"

if st.session_state.uploaded_file:
    with st.spinner('Wait for files uploading...'):
        st.session_state.dfs = load_input_file(st.session_state.uploaded_file)
    st.success('Uploading files done!')

    if st.session_state.dfs is not None:
        st.markdown("### Data preview")
        st.dataframe(pd.DataFrame({'Tables': st.session_state.dfs.keys()}))

        print(222)
        st.markdown("### Preprocessing")
        with st.spinner('Wait for preprocessing...'):
            st.session_state.inputs = preprocessing(st.session_state.dfs)
        st.success('Preprocessing done!')

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
            scenario['num_dcs_open'] = st.number_input('10. Number of Open DCs', min_value = 1, max_value = len(st.session_state.inputs['dcs']), step = 1, value = len(st.session_state.inputs['dcs']))
            scenario['num_factories_open'] = st.number_input('11. Number of Open Factories', min_value = 1, max_value = len(st.session_state.inputs['factories']), step = 1, value = len(st.session_state.inputs['factories']))
            scenario['hist_fc'] = st.checkbox('12. Historical Customer Flows Constraint')
            scenario['hist_ff'] = st.checkbox('13. Historical Intersite Flows Constraint')
            scenario['hist_prod'] = st.checkbox('14. Historical Production Flows Constraint')
            scenario['hist_no_prod'] = st.checkbox('15. Historical Non-Production Flows Constraint')

            submitted = st.form_submit_button(label="Submit")
    
            if submitted:
                st.markdown("### Run Network Optimization model")

                print(scenario)
                if not scenario:
                    print(111111)
                    st.warning("Please select and submit scenario parameters.")
                    st.stop()

    #     
    # with st.spinner('Wait for solving network optimization model...'):
    #     costs, outputs = run_NO_model(inputs, scenario)
    # st.success('Done!')       