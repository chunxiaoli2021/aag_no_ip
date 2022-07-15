### load package
import pandas as pd
import numpy as np
import gurobipy as gp
import time
from datetime import date


### import files
def load_input_file(input_file):
    ### this function is used to import input tables from model input file
    xls = pd.ExcelFile(input_file)
    dfs = {}

    dfs['periods_df'] = pd.read_excel(xls, 'Input_PeriodList')

    dfs['customers_df'] = pd.read_excel(xls, 'Input_CustomerList')
    ### clean customers min perc demand satisfied: cap at 100% & set null to 100% by default 
    dfs['customers_df'].loc[dfs['customers_df']['MinimumPercDemandSatisfied']>1, 'MinimumPercDemandSatisfied'] = 1
    dfs['customers_df'].loc[dfs['customers_df']['MinimumPercDemandSatisfied'].isnull(), 'MinimumPercDemandSatisfied'] = 1

    dfs['sites_df'] = pd.read_excel(xls, 'Input_SiteList')

    dfs['productionlines_df'] = pd.read_excel(xls, 'Input_ProductionLineList')

    dfs['products_df'] = pd.read_excel(xls, 'Input_ProductList')

    dfs['demand_df'] = pd.read_excel(xls, 'Input_CustomerDemand')

    dfs['trans_policies_df'] = pd.read_excel(xls, 'Input_TransportationPolicy')

    dfs['prod_policies_df'] = pd.read_excel(xls, 'Input_ProductionPolicy')

    dfs['inv_df'] = pd.read_excel(xls, 'Input_InventoryPolicy')

    ### load baseline data tables
    dfs['hist_outbound_trans_df'] = pd.read_excel(xls, sheet_name='Input_HistCustomerFlows')

    dfs['hist_inbound_trans_df'] = pd.read_excel(xls, sheet_name='Input_HistIntersiteFlows')

    dfs['hist_prod_df'] = pd.read_excel(xls, sheet_name='Input_HistProduction')

    xls.close()

    return dfs


def preprocessing(dfs):
    ### this function is used to prepare input data lists and dictionaries for the NO model
    inputs = {}

    ### create lists of basic model elements
    inputs['customers'] = [x for x in dfs['customers_df']['CustomerName']]

    inputs['dcs'] = [x for x in dfs['sites_df']['SiteName'][dfs['sites_df']['SiteName'].str.contains('DC')]]

    inputs['factories'] = [x for x in dfs['sites_df']['SiteName'][dfs['sites_df']['SiteName'].str.contains('FAC')]]

    inputs['productionlines'] = [(x['FactoryName'],x['ProductionLine']) for _, x in dfs['productionlines_df'].iterrows()]

    inputs['products'] = [x for x in dfs['products_df']['ProductName']]

    inputs['periods'] = [x for x in dfs['periods_df']['PeriodName']]

    inputs['n_periods'] = len(inputs['periods'])

    ### product type
    inputs['prod_type'] = dfs['products_df'].groupby(['ProductName'])['ProductType'].max().to_dict()

    ### customer demand
    inputs['cust_demand'] = dfs['demand_df'].groupby(['CustomerName', 'ProductName', 'Period'])['CustomerDemand'].sum().to_dict()
    inputs['overall_cust_demand'] = dfs['demand_df'].groupby(['CustomerName'])['CustomerDemand'].sum().to_dict()

    ### customer demand min percsatisfied
    inputs['min_perc_dmd_satisfied'] = dfs['customers_df'].groupby(['CustomerName'])['MinimumPercDemandSatisfied'].max().to_dict()

    ### production capacity
    inputs['line_product_rate'] = dfs['prod_policies_df'].groupby(['FactoryName', 'ProductionLine', 'ProductName'])['MachineHoursPerUnit'].sum().to_dict()

    inputs['line_capacity_cap'] = dfs['productionlines_df'].groupby(['FactoryName', 'ProductionLine'])['MachineHourCapacityPerPeriod'].sum().to_dict()

    inputs['line_product_dict'] = dfs['prod_policies_df'].groupby(['FactoryName','ProductionLine'])['ProductName'].apply(list).to_dict()

    ### factory-attached warehouse storage capacity
    inputs['site_storage_cap'] = dfs['sites_df'][dfs['sites_df']['StorageCapacity'].notnull()].groupby(['SiteName'])['StorageCapacity'].max().to_dict()

    ### factory-attached warehouse handling capacity
    inputs['site_handling_cap'] = dfs['sites_df'][dfs['sites_df']['HandlingCapacityPerPeriod'].notnull()].groupby(['SiteName'])['HandlingCapacityPerPeriod'].max().to_dict()

    ### factory variable production cost
    inputs['var_prod_cost'] = dfs['prod_policies_df'].groupby(['FactoryName', 'ProductionLine', 'ProductName'])['VariableProductionCost'].sum().to_dict()

    ### dc variable handling cost 
    inputs['var_handling_cost'] = dfs['sites_df'].groupby(['SiteName'])['VariableHandlingCost'].sum().to_dict()

    ### factory fixed cost
    inputs['site_fixed_cost'] = dfs['sites_df'].groupby(['SiteName'])['FixedOperatingCostPerHorizon'].sum().to_dict()

    ### product line fixed cost
    inputs['productline_fixed_cost'] = dfs['productionlines_df'].groupby(['FactoryName','ProductionLine'])['FixedOperatingCostPerHorizon'].sum().to_dict()

    ### transportation cost
    inputs['transp_cost'] = dfs['trans_policies_df'].groupby(['Origin', 'Destination', 'ProductType'])['VariableTransportationCost'].max().to_dict()

    ### transportation distance
    inputs['transp_dist'] = dfs['trans_policies_df'].groupby(['Origin', 'Destination', 'ProductType'])['Distance'].max().to_dict()

    ### unit product cubic meters
    inputs['product_cubic'] = dfs['products_df'].groupby(['ProductName'])['Volume'].max().to_dict()

    ### inventory turns
    inputs['inv_turns'] = dfs['inv_df'].groupby(['SiteName','ProductName'])['InventoryTurns'].max().to_dict()

    ### initial inventory
    inputs['initial_inv'] = dfs['inv_df'].groupby(['SiteName','ProductName'])['InitialInventory'].max().to_dict()

    ### generate index sets
    inputs['ss_lanes'] = gp.tuplelist([(o, d, p, m) for o in inputs['factories'] for d in inputs['dcs'] for p in inputs['products'] for m in inputs['periods']]) 

    inputs['sc_lanes'] = gp.tuplelist([(o, d, p, m) for o in inputs['dcs'] for (d, p, m) in inputs['cust_demand']]) 

    inputs['prod_policies'] = gp.tuplelist([(f, l, p, m) for (f, l, p) in inputs['line_product_rate'] for m in inputs['periods']])


    inputs['dc_inv_basis'] = gp.tuplelist([(dc, p, m) for dc in inputs['dcs'] for p in inputs['products'] for m in list(inputs['periods']) + [max(inputs['periods'])+1]])

    inputs['f_inv_basis'] = gp.tuplelist([(f, p, m) for f in inputs['factories'] for p in inputs['products'] for m in list(inputs['periods']) + [max(inputs['periods'])+1]])

    ### per unit penalty cost of using dummy lanes
    inputs['dmd_pen'] = np.ceil(max(inputs['var_prod_cost'].values())*10 + max(inputs['var_handling_cost'].values())*10 + max(inputs['transp_cost'].values())*10)

    return inputs


def run_NO_model(inputs, scenario):
    ####################### build no model
    print("----------------------Building Network Optimization Model----------------------")
    time_start = time.time()

    ### create model
    m = gp.Model('BXNetworkOpt')

    ### decision variables
    ss_flow = m.addVars(inputs['ss_lanes'], vtype=gp.GRB.CONTINUOUS, name='ff_flow') # factory to dc flows
    sc_flow = m.addVars(inputs['sc_lanes'], vtype=gp.GRB.CONTINUOUS, name='fc_flow') # dc to customer flows
    prod_flow = m.addVars(inputs['prod_policies'], vtype=gp.GRB.CONTINUOUS, name='prod_flow') # production volume
    f_inv_level = m.addVars(inputs['f_inv_basis'], vtype=gp.GRB.CONTINUOUS, name='f_inv_level') # factory beginning inventory level of each period
    dc_inv_level = m.addVars(inputs['dc_inv_basis'], vtype=gp.GRB.CONTINUOUS, name='dc_inv_level') # dc beginning inventory level of each period
    f_open = m.addVars(inputs['factories'], vtype=gp.GRB.BINARY, name='f_open') # factory open or not
    dc_open = m.addVars(inputs['dcs'], vtype=gp.GRB.BINARY, name='dc_open') # dc open or not
    productline_open = m.addVars(inputs['productionlines'], vtype=gp.GRB.BINARY, name='dc_open') # product line open or not
    demand_slack = m.addVars(inputs['cust_demand'], vtype=gp.GRB.CONTINUOUS, name='demand_slack') # slack variable for unmatched demand

    m.update()

    ### objective function
    tot_var_prod_cost = gp.quicksum(prod_flow[(f, l, p, m)]*inputs['var_prod_cost'][(f, l, p)] for (f, l, p, m) in inputs['prod_policies'])
    tot_dc_var_handling_cost = gp.quicksum(sc_flow[(o, d, p, m)]*inputs['var_handling_cost'][(o)] for (o, d, p, m) in inputs['sc_lanes'])
    tot_f_var_handling_cost = gp.quicksum(ss_flow[(o, d, p, m)]*inputs['var_handling_cost'][(o)] for (o, d, p, m) in inputs['ss_lanes']) 
    tot_site_fixed_cost = gp.quicksum(f_open[f]*inputs['site_fixed_cost'][f] for f in inputs['factories']) + \
                            gp.quicksum(dc_open[dc]*inputs['site_fixed_cost'][dc] for dc in inputs['dcs']) + \
                            gp.quicksum(productline_open[(f, l)]*inputs['productline_fixed_cost'][(f, l)] for (f, l) in inputs['productionlines'])
    tot_ss_transp_cost = gp.quicksum(ss_flow[(o, d, p, m)]*inputs['product_cubic'][p]*inputs['transp_cost'][(o, d, inputs['prod_type'][p])] for (o, d, p, m) in inputs['ss_lanes'])
    tot_sc_transp_cost = gp.quicksum(sc_flow[(o, d, p, m)]*inputs['product_cubic'][p]*inputs['transp_cost'][(o, d, inputs['prod_type'][p])] for (o, d, p, m) in inputs['sc_lanes'])
    tot_dmd_pen_cost = demand_slack.sum('*', '*', '*')*inputs['dmd_pen']
    tot_cost = tot_var_prod_cost + tot_dc_var_handling_cost + tot_f_var_handling_cost + tot_site_fixed_cost + tot_ss_transp_cost + tot_sc_transp_cost + tot_dmd_pen_cost

    m.setObjective(tot_cost, gp.GRB.MINIMIZE)

    ### demand satisfaction
    ### soft demand satisfication cons
    m.addConstrs(
        (sc_flow.sum('*', c, p, m) == inputs['cust_demand'][(c, p, m)] + demand_slack[(c, p, m)] for (c, p, m) in inputs['cust_demand'].keys()), 'customer_demand'
    )

    ### at least x% demand must be satisfied for each customer
    if scenario['perc_demand_satisfied']:
        m.addConstr(
            (sc_flow.sum('*', c, '*', '*') >= 0.8 * sum(inputs['overall_cust_demand'][c]) for c in inputs['customers']), 'perc_demand_satisfied'
        )

    ### flow balance
    ### factory flow balance
    m.addConstrs(
        (f_inv_level[(f, p, m)] + prod_flow.sum(f, '*', p, m) == ss_flow.sum(f, '*', p, m) + f_inv_level[(f, p, m+1)] for f in inputs['factories'] for p in inputs['products'] for m in inputs['periods']), 'flow_balance'
    )
    ### dc flow balance
    m.addConstrs(
        (dc_inv_level[(dc, p, m)] + ss_flow.sum('*', dc, p, m) == sc_flow.sum(dc, '*', p, m) + dc_inv_level[(dc, p, m+1)]
        for dc in inputs['dcs'] for p in inputs['products'] for m in inputs['periods']), 'flow_balance'
    )

    ### factory production capacity by line
    m.addConstrs(
        (gp.quicksum([prod_flow[(f, l, p, m)] * inputs['line_product_rate'][(f, l, p)] for p in inputs['line_product_dict'][(f, l)]])<= inputs['line_capacity_cap'][(f, l)]*productline_open[(f, l)] for (f, l) in inputs['productionlines'] for m in inputs['periods']), 'line_capacity_cap'
    )

    ### product line open
    m.addConstrs(
        (productline_open[(f, l)] <= f_open[f] for (f, l) in inputs['productionlines']), 'productline_open'
    )

    ### dc handling capacity
    if scenario['dc_handling_cap']:
        m.addConstrs(
            (sc_flow.sum(dc, '*', '*', m) 
            <= inputs['site_handling_cap'][dc]*dc_open[dc] for dc in set(inputs['dcs']).intersection(inputs['site_handling_cap'].keys()) for m in inputs['periods']), 'dc_handling_cap'
        )

    ### dc storage capacity
    if scenario['dc_storage_cap']:
        m.addConstrs(
            (dc_inv_level.sum(dc, '*', m) 
            <= inputs['site_storage_cap'][dc]*dc_open[dc] for dc in set(inputs['dcs']).intersection(inputs['site_storage_cap'].keys()) for m in inputs['periods']), 'dc_storage_cap'
        )

    ### factory handling capacity
    if scenario['factory_handling_cap']:
        m.addConstrs(
            (ss_flow.sum(f, '*', '*', m) 
            <= inputs['site_handling_cap'][f]*f_open[f] for f in set(inputs['factories']).intersection(inputs['site_handling_cap'].keys()) for m in inputs['periods']), 'factory_handling_cap'
        )

    ### factory storage capacity
    if scenario['factory_storage_cap']:
        m.addConstrs(
            (f_inv_level.sum(f, '*', m) 
            <= inputs['site_storage_cap'][f]*f_open[f] for f in set(inputs['factories']).intersection(inputs['site_storage_cap'].keys()) for m in inputs['periods']), 'factory_storage_cap'
        )



    ### number of factories open
    if scenario['fix_factories_open']:
        m.addConstr(
            (f_open.sum('*') == scenario['num_factories_open']), 'num_factories_open'
        )

    ### number of dcs open
    if scenario['fix_dcs_open']:
        m.addConstr(
            (dc_open.sum('*') == scenario['num_dcs_open']), 'num_dcs_open'
        )

    ### dc inital inventory
    if scenario['dc_initial_inv']:
        m.addConstrs(
            (dc_inv_level[(dc, p, 1)] == inputs['initial_inv'][(dc, p)] for dc in inputs['dcs'] for p in inputs['products']), 'dc_intial_inv'
        )

    ### factory inital inventory
    if scenario['factory_initial_inv']:
        m.addConstrs(
            (f_inv_level[(f, p, 1)] == inputs['initial_inv'][(f, p)] for f in inputs['factories'] for p in inputs['products']), 'f_intial_inv'
        )

    ##########################historical cons
    ### historical factory-to-customer flows
    if scenario['hist_fc']:
        m.addConstrs(
            (sc_flow[(o, d, p, m)] == inputs['hist_fc_flows'][(o, d, p, m)] for (o, d, p, m) in inputs['hist_fc_flows'].keys()), 'hist_fc'
        )

    ### historical inter-factory flows
    if scenario['hist_ff']:
        m.addConstrs(
            (ss_flow[(o, d, p, m)] == inputs['hist_ff_flows'][(o, d, p, m)] for (o, d, p, m) in inputs['hist_ff_flows'].keys()), 'hist_ff'
        )

    ### historical production
    if scenario['hist_prod']:
        m.addConstrs(
            (prod_flow[(f, l, p, m)] == inputs['hist_prod_flows'][(f, l, p, m)] for (f, l, p, m) in inputs['hist_prod_flows'].keys()), 'hist_prod'
        )

    ### historical no production
    if scenario['hist_no_prod']:
        m.addConstrs(
            (prod_flow[(f, l, p, m)] == 0 for (f, l, p, m) in [x for x in inputs['prod_policies'] if x not in inputs['hist_prod_flows'].keys()]), 
            'hist_no_prod'
        )

    time_build_end = time.time()

    ### solve the model
    print("----------------------Solving Network Optimization Model----------------------")
    m.optimize()
    time_solve_end = time.time()

    print()
    print("----------------------Output Network Optimization Results----------------------")
    print('optimal' if m.status==gp.GRB.OPTIMAL else 'infeasible')
    print('model build time:', round(time_build_end - time_start), 's')
    print('model solve time:', round(time_solve_end - time_build_end), 's')
    tot_oprt_cost = tot_var_prod_cost.getValue() + tot_dc_var_handling_cost.getValue() + tot_f_var_handling_cost.getValue() \
                    + tot_site_fixed_cost.getValue() + tot_sc_transp_cost.getValue() + tot_ss_transp_cost.getValue()
   

    ### output costs
    costs = {
        'tot_site_fixed_cost': tot_site_fixed_cost.getValue(),
        'tot_var_prod_cost': tot_var_prod_cost.getValue(),
        'tot_dc_var_handling_cost': tot_dc_var_handling_cost.getValue(),
        'tot_f_var_handling_cost': tot_f_var_handling_cost.getValue(),
        'tot_sc_transp_cost': tot_sc_transp_cost.getValue(),
        'tot_ss_transp_cost': tot_ss_transp_cost.getValue(),
        'tot_dmd_pen_cost': tot_dmd_pen_cost.getValue(),
        'tot_oprt_cost': tot_oprt_cost,
        'tot_cost': tot_cost.getValue()
    }
    print('Total operating cost:', tot_oprt_cost)
    
    ### output decision variables
    outputs = {}
    
    ### site to customer flows
    sc_flow_res = []

    for i in inputs['sc_lanes']:
        if sc_flow[i].x > 0:
            var_output = {
                'Origin': i[0],
                'Destination': i[1],
                'ProductName': i[2],
                'ProductType': inputs['prod_type'][i[2]],
                'Period': i[3],
                'Quantity': sc_flow[i].x
            }
            sc_flow_res.append(var_output)

    outputs['sc_flow_res_df'] = pd.DataFrame.from_records(sc_flow_res)

    ### site to site flows
    ss_flow_res = []

    for i in inputs['ss_lanes']:
        if ss_flow[i].x > 0:
            var_output = {
                'Origin': i[0],
                'Destination': i[1],
                'ProductName': i[2],
                'ProductType': inputs['prod_type'][i[2]],
                'Period': i[3],
                'Quantity': ss_flow[i].x
            }
            ss_flow_res.append(var_output)

    outputs['ss_flow_res_df'] = pd.DataFrame.from_records(ss_flow_res)

    ### production flows
    prod_flow_res = []

    for i in inputs['prod_policies']:
        if prod_flow[i].x > 0:
            var_output = {
                'FactoryName': i[0],
                'ProductionLine': i[1],
                'ProductName': i[2],
                'ProductType': inputs['prod_type'][i[2]],
                'Period': i[3],
                'Quantity': prod_flow[i].x
            }
            prod_flow_res.append(var_output)
            
    outputs['prod_flow_res_df'] = pd.DataFrame.from_records(prod_flow_res)

    ### DC inventory level
    dc_inv_level_res = []

    for i in inputs['dc_inv_basis']:
        var_output = {
            'SiteName': i[0],
            'SiteType': 'Distribution Center',
            'ProductName': i[1],
            'ProductType': inputs['prod_type'][i[1]],
            'Period': i[2],
            'Quantity': dc_inv_level[i].x
        }
        dc_inv_level_res.append(var_output)
        
    outputs['dc_inv_level_res_df'] = pd.DataFrame.from_records(dc_inv_level_res)

    ### factory inventory level
    f_inv_level_res = []

    for i in inputs['f_inv_basis']:
        var_output = {
            'SiteName': i[0],
            'SiteType': 'Factory',
            'ProductName': i[1],
            'ProductType': inputs['prod_type'][i[1]],
            'Period': i[2],
            'Quantity': f_inv_level[i].x
        }
        f_inv_level_res.append(var_output)
            
    outputs['f_inv_level_res_df'] = pd.DataFrame.from_records(f_inv_level_res)

    ### concat factory and dc inventory level df
    outputs['inv_level_res_df'] = pd.concat([outputs['f_inv_level_res_df'], outputs['dc_inv_level_res_df']])
    outputs['inv_level_res_df'] = outputs['inv_level_res_df'][outputs['inv_level_res_df']['Period']!=1]\
                        .assign(Period = lambda x: x['Period']-1)\
                            .rename({'Quantity' : 'EndOfPeriodInventory'}, axis=1)


    ### demand slack
    demand_slack_res = [] 

    for i in inputs['cust_demand']:
        if demand_slack[i].x > 0:
            var_output = {
                'FactoryName': i[0],
                'ProductName': i[1],
                'ProductType': inputs['prod_type'][i[1]],
                'Period': i[2],
                'Quantity': demand_slack[i].x
            }
            demand_slack_res.append(var_output)
            
    outputs['demand_slack_res_df'] = pd.DataFrame.from_records(demand_slack_res)

    if len(outputs['demand_slack_res_df'])>1:
        print(outputs['demand_slack_res_df'][['Quantity']].sum())
    else:
        print('no unsatisfied demand')


    ### factory open
    f_open_res = []

    for i in inputs['factories']:
        var_output = {
            'SiteName': i,
            'SiteType': 'Factory',
            'SiteStatus': f_open[i].x
        }
        f_open_res.append(var_output)
            
    outputs['f_open_res_df'] = pd.DataFrame.from_records(f_open_res)

    ### DC open
    dc_open_res = []

    for i in inputs['dcs']:
        var_output = {
            'SiteName': i,
            'SiteType': 'Distribution Center',
            'SiteStatus': dc_open[i].x
        }
        dc_open_res.append(var_output)
            
    outputs['dc_open_res_df'] = pd.DataFrame.from_records(dc_open_res)

    ### production line open
    productline_open_res = []

    for i in inputs['productionlines']:
        var_output = {
            'FactoryName': i[0],
            'ProductionLine': i[1],
            'LineStatus': productline_open[i].x
        }
        productline_open_res.append(var_output)
            
    outputs['productline_open_res_df'] = pd.DataFrame.from_records(productline_open_res)

    return costs, outputs

def postprocessing(dfs, inputs, outputs):
    res_dfs = {}

    ### add cost of flow and production
    res_dfs['sc_flow_res_df_1'] = outputs['sc_flow_res_df'].merge(dfs['trans_policies_df'], on = ['Origin','Destination','ProductType'])\
        .merge(dfs['products_df'][['ProductName','Volume']], on = ['ProductName'])\
            .assign(**{"TransportationCost": lambda x: x['Quantity']*x['Volume']*x['VariableTransportationCost']})\
                .merge(dfs['sites_df'][['SiteName','VariableHandlingCost']].rename({'SiteName':'Origin'},axis=1), on = ['Origin'])\
                    .assign(**{"HandlingCost": lambda x: x['Quantity']*x['VariableHandlingCost']})


    res_dfs['ss_flow_res_df_1'] = outputs['ss_flow_res_df'].merge(dfs['trans_policies_df'], on = ['Origin','Destination','ProductType'])\
        .merge(dfs['products_df'][['ProductName','Volume']], on = ['ProductName'])\
            .assign(**{"TransportationCost": lambda x: x['Quantity']*x['Volume']*x['VariableTransportationCost']})\
                .merge(dfs['sites_df'][['SiteName','VariableHandlingCost']].rename({'SiteName':'Origin'},axis=1), on = ['Origin'])\
                    .assign(**{"HandlingCost": lambda x: x['Quantity']*x['VariableHandlingCost']})

    res_dfs['prod_flow_res_df_1'] = outputs['prod_flow_res_df'].merge(dfs['prod_policies_df'], on = ['FactoryName','ProductionLine','ProductName'])\
        .assign(**{"ProductionCost": lambda x: x['Quantity']*x['VariableProductionCost']})\
            .assign(**{'MachineHours': lambda x: x['Quantity'] * x['MachineHoursPerUnit']})

    res_dfs['site_open_res_df_1'] = dfs['sites_df'][['SiteName','FixedOperatingCostPerHorizon']].merge(pd.concat([outputs['f_open_res_df'],outputs['dc_open_res_df']]), on = 'SiteName', how= 'left')\
        .assign(**{'SiteFixedOperatingCost': lambda x: x['SiteStatus']*x['FixedOperatingCostPerHorizon']})

    res_dfs['inv_level_res_df_1'] = outputs['inv_level_res_df'].merge(pd.concat([outputs['ss_flow_res_df'].groupby(['Origin','ProductName','Period'])['Quantity'].sum().reset_index().rename({'Origin':'SiteName','Quantity':'Throughput'}, axis=1),
                                    outputs['sc_flow_res_df'].groupby(['Origin','ProductName','Period'])['Quantity'].sum().reset_index().rename({'Origin':'SiteName','Quantity':'Throughput'}, axis=1)]),
                                    on = ['SiteName', 'ProductName', 'Period'])\
                                        .merge(dfs['inv_df'][['SiteName', 'ProductName','InventoryTurns']], on = ['SiteName', 'ProductName'])\
                                            .assign(MonthlyInventoryTurns = lambda x: x['InventoryTurns'] / 12,
                                                    TurnEstimatedInventory  = lambda x: x['Throughput'] / x['InventoryTurns'])\
                                                        .drop(['InventoryTurns'], axis=1)
                                                        
    ### period cost summary 
    res_dfs['period_cost_summary'] = res_dfs['sc_flow_res_df_1'].groupby(['Period'])[['TransportationCost','HandlingCost','Quantity']].sum().reset_index()\
        .rename({'TransportationCost':'TransportationCostCustomerFlows','HandlingCost':'DCHandlingCost','Quantity':'QuantityCustomerFlows'},axis=1)\
            .merge(res_dfs['ss_flow_res_df_1'].groupby(['Period'])[['TransportationCost','HandlingCost','Quantity']].sum().reset_index()\
                .rename({'TransportationCost':'TransportationCostIntersiteFlows','HandlingCost':'FactoryHandlingCost','Quantity':'QuantityIntersiteFlows'},axis=1), on = ['Period'])\
                    .assign(**{'TransportationCost': lambda x: (x['TransportationCostCustomerFlows']+x['TransportationCostIntersiteFlows'])})\
                        .merge(res_dfs['prod_flow_res_df_1'].groupby(['Period'])[['ProductionCost','Quantity','MachineHours']].sum().reset_index()\
                            .rename({'ProductionCost':'FactoryProductionCost', 'Quantity':'FactoryProductionQuantity'},axis=1), on = ['Period'])\
                                .assign(SiteFixedOperatingCost = res_dfs['site_open_res_df_1'][['FixedOperatingCostPerHorizon']].sum()[0]/inputs['n_periods'], 
                                        NumberOfSitesOpen = res_dfs['site_open_res_df_1'][['SiteStatus']].sum()[0])\
                                            .assign(TotalCost = lambda x: x['TransportationCost']+x['DCHandlingCost']+x['FactoryProductionCost']+x['SiteFixedOperatingCost'])\
                                                .merge(dfs['demand_df'].groupby('Period')['CustomerDemand'].sum().reset_index().rename({'CustomerDemand':'TotalDemand'}, axis=1), on = ['Period'])\
                                                    .assign(DemandSatisfiedPerc = lambda x: x['QuantityCustomerFlows'] / x['TotalDemand'])
                                               
    ### site cost summary
    ##### factory cost summary
    res_dfs['factory_cost_summary'] = res_dfs['ss_flow_res_df_1'].groupby(['Origin']).apply(lambda x : pd.Series({'TransportationCost': x['TransportationCost'].sum(),
                                                                                            'HandlingCost': x['HandlingCost'].sum(),
                                                                                            'Quantity': x['Quantity'].sum(),
                                                                                            'MaxDistance':x['Distance'].max(),
                                                                                            'WeightedAverageDistance':(x['Quantity']*x['Volume']*x['Distance']).sum() / (x['Quantity']*x['Volume']).sum(),
                                                                                            })).reset_index()\
                .rename({'Origin':'SiteName', 'TransportationCost':'OutboundTransportationCost', 'Quantity':'ThoughputLevel'},axis=1)\
                    .merge(outputs['f_open_res_df'].merge(dfs['sites_df'][['SiteName','FixedOperatingCostPerHorizon']], on = ['SiteName']).assign(FixedOperatingCostPerHorizon = lambda x: x['FixedOperatingCostPerHorizon']*x['SiteStatus']), on = ['SiteName'])\
                        .merge(res_dfs['prod_flow_res_df_1'].groupby(['FactoryName'])[['ProductionCost','MachineHours']].sum().reset_index()\
                            .merge(dfs['productionlines_df'].groupby(['FactoryName']).agg({'MachineHourCapacityPerPeriod':'sum'}).reset_index(), on = ['FactoryName'])\
                                .assign(MachineHourCapacityPerHorizon = lambda x: x['MachineHourCapacityPerPeriod']*inputs['n_periods'],
                                        ProductionCapacityUtilization = lambda x: x['MachineHours']/x['MachineHourCapacityPerHorizon'])\
                                        .rename({'FactoryName':'SiteName'},axis=1), on = 'SiteName')\
                                            .merge(dfs['sites_df'][['SiteName','StorageCapacity','HandlingCapacityPerPeriod']], on = ['SiteName'])\
                                                .assign(HandlingCapacityPerHorizon = lambda x: x['HandlingCapacityPerPeriod']*inputs['n_periods'])\
                                                    .merge(res_dfs['inv_level_res_df_1'].groupby(['SiteName'])[['TurnEstimatedInventory','EndOfPeriodInventory']].mean().reset_index(), on = ['SiteName'])\
                                                        .assign(HandlingCapacityUtilization = lambda x: x['ThoughputLevel']/x['HandlingCapacityPerHorizon'],
                                                                StorageCapacityUtilization = lambda x: np.where(x['StorageCapacity'] == 0, np.NaN, x['EndOfPeriodInventory']/x['StorageCapacity']))\
                                                                    .drop(['MachineHourCapacityPerPeriod','MachineHourCapacityPerHorizon','MachineHours'], axis=1)
                                            
    ##### dc cost summary
    res_dfs['dc_cost_summary'] = res_dfs['sc_flow_res_df_1'].groupby(['Origin']).apply(lambda x : pd.Series({'TransportationCost': x['TransportationCost'].sum(),
                                                                                        'HandlingCost': x['HandlingCost'].sum(),
                                                                                        'Quantity': x['Quantity'].sum(),
                                                                                        'MaxDistance':x['Distance'].max(),
                                                                                        'WeightedAverageDistance':(x['Quantity']*x['Volume']*x['Distance']).sum() / (x['Quantity']*x['Volume']).sum(),
                                                                                        })).reset_index()\
        .rename({'Origin':'SiteName', 'TransportationCost':'OutboundTransportationCost', 'Quantity':'ThoughputLevel'},axis=1)\
            .merge(outputs['dc_open_res_df'].merge(dfs['sites_df'][['SiteName','FixedOperatingCostPerHorizon']], on = ['SiteName']).assign(FixedOperatingCostPerHorizon = lambda x: x['FixedOperatingCostPerHorizon']*x['SiteStatus']), on = ['SiteName'])\
                .merge(dfs['sites_df'][['SiteName','StorageCapacity','HandlingCapacityPerPeriod']], on = ['SiteName'])\
                    .assign(HandlingCapacityPerHorizon = lambda x: x['HandlingCapacityPerPeriod']*inputs['n_periods'])\
                        .merge(res_dfs['inv_level_res_df_1'].groupby(['SiteName'])[['TurnEstimatedInventory','EndOfPeriodInventory']].mean().reset_index(), on = ['SiteName'])\
                            .assign(HandlingCapacityUtilization = lambda x: x['ThoughputLevel']/x['HandlingCapacityPerHorizon'],
                                    StorageCapacityUtilization = lambda x: np.where(x['StorageCapacity'] == 0, np.NaN, x['EndOfPeriodInventory']/x['StorageCapacity']))
                
    res_dfs['site_cost_summary'] = pd.concat([res_dfs['factory_cost_summary'], res_dfs['dc_cost_summary']]).drop(['HandlingCapacityPerPeriod'], axis=1)

    ### product line cost summary
    res_dfs['productline_cost_summary'] = res_dfs['prod_flow_res_df_1'].groupby(['FactoryName','ProductionLine'])[['ProductionCost','MachineHours']].sum().reset_index()\
        .merge(outputs['productline_open_res_df'], on = ['FactoryName','ProductionLine'], how = 'outer').fillna(0) \
            .merge(dfs['productionlines_df'][['FactoryName','ProductionLine','MachineHourCapacityPerPeriod','FixedOperatingCostPerHorizon']].assign(), on = ['FactoryName','ProductionLine'], how = 'outer')\
                .assign(FixedOperatingCost = lambda x: x['FixedOperatingCostPerHorizon'] * x['LineStatus'],
                        MachineHourCapacityPerHorizon = lambda x: x['MachineHourCapacityPerPeriod']*inputs['n_periods'],
                        ProductionCapacityUtilization = lambda x: x['MachineHours']/x['MachineHourCapacityPerHorizon'])

    return res_dfs

def save_output_to_excel(output_file, scenario, costs, dfs, res_dfs):
    ### save model output template
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

    ### output data tables
    pd.DataFrame({'Scenario': [scenario['name']],
                    'TotalCost':[costs['tot_cost']],
                    'SiteFixedOperatingCost':[costs['tot_site_fixed_cost']],
                    'FactoryProductionCost':[costs['tot_var_prod_cost']],
                    'FactoryHandlingCost':[costs['tot_f_var_handling_cost']],
                    'DCHandlingCost':[costs['tot_dc_var_handling_cost']],
                    'TransportationCost':[costs['tot_sc_transp_cost']+costs['tot_ss_transp_cost']],
                    'TransportationCostCustomerFlows':[costs['tot_sc_transp_cost']],
                    'TransportationCostIntersiteFlows':[costs['tot_ss_transp_cost']],
                    'DemandSatisfiedPerc': [res_dfs['sc_flow_res_df_1']['Quantity'].sum() / dfs['demand_df']['CustomerDemand'].sum()],
                    'TotalDemand':[dfs['demand_df']['CustomerDemand'].sum()],
                    'QuantityCustomerFlows':[res_dfs['sc_flow_res_df_1']['Quantity'].sum()],
                    'QuantityIntersiteFlows':[res_dfs['ss_flow_res_df_1']['Quantity'].sum()],
                    'FactoryProductionQuantity':[res_dfs['prod_flow_res_df_1']['Quantity'].sum()],
                    'NumberOfDCsOpen':[res_dfs['site_open_res_df_1'].loc[res_dfs['site_open_res_df_1']['SiteType']=='Distribution Center','SiteStatus'].sum()],
                    'NumberOfFactoriesOpen':[res_dfs['site_open_res_df_1'].loc[res_dfs['site_open_res_df_1']['SiteType']=='Factory','SiteStatus'].sum()],
                    'MaxDistanceCustomerFlows':[res_dfs['sc_flow_res_df_1']['Distance'].max()],
                    'WeightedAverageDistanceCustomerFlows':[(res_dfs['sc_flow_res_df_1']['Quantity']*res_dfs['sc_flow_res_df_1']['Volume']*res_dfs['sc_flow_res_df_1']['Distance']).sum() / (res_dfs['sc_flow_res_df_1']['Quantity']*res_dfs['sc_flow_res_df_1']['Volume']).sum()],
                    'WeightedAverageDistanceIntersiteFlows':[(res_dfs['ss_flow_res_df_1']['Quantity']*res_dfs['ss_flow_res_df_1']['Volume']*res_dfs['ss_flow_res_df_1']['Distance']).sum() / (res_dfs['ss_flow_res_df_1']['Quantity']*res_dfs['ss_flow_res_df_1']['Volume']).sum()]
                    })\
            .to_excel(writer, sheet_name='Output_CostSummary', index=False)
    res_dfs['period_cost_summary'].assign(**{'Scenario': scenario['name']})[['Scenario','Period','TotalCost','SiteFixedOperatingCost','FactoryProductionCost','FactoryHandlingCost',
                            'DCHandlingCost','TransportationCost','TransportationCostCustomerFlows','TransportationCostIntersiteFlows',
                            'DemandSatisfiedPerc','TotalDemand','QuantityCustomerFlows','QuantityIntersiteFlows','FactoryProductionQuantity']] \
                            .to_excel(writer, sheet_name='Output_CostSummaryByPeriod', index=False)
    #site_cost_summary.insert(0, 'Scenario', scenario)
    res_dfs['site_cost_summary'].assign(**{'Scenario': scenario['name']})[['Scenario','SiteName', 'SiteType','SiteStatus', 'FixedOperatingCostPerHorizon','ProductionCost','HandlingCost','OutboundTransportationCost',
                    'ProductionCapacityUtilization','ThoughputLevel', 'HandlingCapacityPerHorizon','HandlingCapacityUtilization','StorageCapacity','TurnEstimatedInventory','EndOfPeriodInventory','StorageCapacityUtilization',
                    'MaxDistance', 'WeightedAverageDistance']]\
                    .to_excel(writer, sheet_name='Output_CostSummaryBySite', index=False)
    res_dfs['productline_cost_summary'].assign(**{'Scenario': scenario['name']})[['Scenario','FactoryName','ProductionLine','LineStatus','FixedOperatingCost','ProductionCost','MachineHours','MachineHourCapacityPerHorizon','ProductionCapacityUtilization']] \
            .to_excel(writer, sheet_name='Output_CostSummaryByProductLine', index=False)
    

    res_dfs['sc_flow_res_df_1'].assign(**{'Scenario': scenario['name']})[['Scenario','Origin','Destination','ProductName','ProductType','Period','Quantity','TransportationCost','HandlingCost']].to_excel(writer, sheet_name='Output_CustomerFlows', index=False)
    res_dfs['ss_flow_res_df_1'].assign(**{'Scenario': scenario['name']})[['Scenario','Origin','Destination','ProductName','ProductType','Period','Quantity','TransportationCost']].to_excel(writer, sheet_name='Output_IntersiteFlows', index=False)
    res_dfs['prod_flow_res_df_1'].assign(**{'Scenario': scenario['name']})[['Scenario','FactoryName', 'ProductionLine', 'ProductName','ProductType', 'Period', 'Quantity', 'ProductionCost']].to_excel(writer, sheet_name='Output_ProductionFlows', index=False)
    res_dfs['inv_level_res_df_1'].assign(**{'Scenario': scenario['name']})[['Scenario','SiteName','ProductName','ProductType','Period','EndOfPeriodInventory','TurnEstimatedInventory']].to_excel(writer, sheet_name='Output_InventoryLevel', index=False)

    ### need further cleaning format before send to client
    ### input data tables
    dfs['periods_df'].to_excel(writer, sheet_name='Input_PeriodList', index=False)
    dfs['products_df'].to_excel(writer, sheet_name='Input_ProductList', index=False)
    dfs['customers_df'].to_excel(writer, sheet_name='Input_CustomerList', index=False)
    dfs['sites_df'].to_excel(writer, sheet_name='Input_SiteList', index=False)
    dfs['productionlines_df'].to_excel(writer, sheet_name='Input_ProductLineList', index=False)
    dfs['trans_policies_df'].to_excel(writer, sheet_name='Input_TransportationPolicy', index=False)
    dfs['prod_policies_df'].to_excel(writer, sheet_name='Input_ProductionPolicy', index=False)
    dfs['inv_df'].to_excel(writer, sheet_name='Input_InventoryPolicy', index=False)
    dfs['demand_df'].to_excel(writer, sheet_name='Input_CustomerDemand', index=False)
    dfs['hist_outbound_trans_df'].to_excel(writer, sheet_name='Input_HistShipmentFactoryToCust', index=False)
    dfs['hist_inbound_trans_df'].to_excel(writer, sheet_name='Input_HistShipmentAmongFactory', index=False)
    dfs['hist_prod_df'].to_excel(writer, sheet_name='Input_HistProduction', index=False)

    writer.save()
    writer.close()

if __name__ == "__main__":
    model_sanitized_path = 'C:/Users/52427/Documents/Cases/Case11_Baixiang/Baixiang_supply_chain_optimization/model_sanitized/'
    input_file = model_sanitized_path + 'model_input_20220622_EN.xlsx'

    ### import files
    dfs = load_input_file(input_file)
    ### preprocess
    inputs = preprocessing(dfs)
    # scenarios matrix
    scenario = {'name': 'Unconstrainted+RemoveHandlingCons',
                'dc_handling_cap': False, 
                'dc_storage_cap': True,
                'factory_handling_cap': False, 
                'factory_storage_cap': True,
                'perc_demand_satisfied': False,
                'hist_fc': False, 
                'hist_ff': False, 
                'hist_prod': False, 
                'hist_no_prod': False, 
                'fix_factories_open': True,
                'fix_dcs_open': True,
                'num_factories_open': 9, 
                'num_dcs_open': 9, 
                'dc_initial_inv': True,
                'factory_initial_inv': True}

    ### run NO model
    costs, outputs = run_NO_model(inputs, scenario)
    ### postprocess
    res_dfs = postprocessing(dfs, inputs, outputs)
    ### save to excel
    today = date.today()
    today = str(today).replace('-','')
    output_file = model_sanitized_path + 'model_output_'+ today +'_'+scenario['name']+'.xlsx'
    save_output_to_excel(output_file, scenario, costs, dfs, res_dfs)
