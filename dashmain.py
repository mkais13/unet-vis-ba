
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import numpy as np
import dash

from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context
import os
import tensorboard as tb

app = Dash(__name__, update_title=None, external_stylesheets=[dbc.themes.FLATLY])

def create_picture_options():
    result = []
    for i in range(30):
        result.append({"label": str(i), "value" : str(i)})
    return result








app.layout = dbc.Container([
dbc.NavbarSimple(color="primary",brand = "SIMILARITY PLOT OF UNET TRAININGS WITH DIFFERENT HYPERPARAMETERS", dark=True, style={"padding-left" : "3vh", "height" : "8vh"}),

dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="dropdown_text", children=["select the id of the picture to be analyzed"], className="text-center"),
                            html.Div([
                                dcc.Dropdown(   
                                    id="selected_picture_id",
                                    options= create_picture_options(), 
                                    multi=False,
                                    value="0",
                                    clearable=False,
                                ),
                            ]),
                        ], align="center"),
                        dbc.Col([

                            html.Div(id="radio_text", children=["select the dimension to analyze in"], className="text-center"),

                            html.Div([
                                dbc.RadioItems(
                                    id="selected_dimension",
                                    class_name="btn-group",
                                    inputClassName="btn-check",
                                    labelClassName="btn btn-outline-primary",
                                    labelCheckedClassName="active",
                                    options=[{"label" :"2D", "value": "2D"}, {"label": "3D","value":"3D"}], 
                                    value="2D",
                                    style={"padding-right" : "3vh"}
                                ),
                            ], className="text-center"),

                        ],align="center"),
                    ], justify= "evenly", align="center"),
                    dcc.Graph(id="similaritygraph", style={"height" : "40vh"}),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                    
                        dbc.Col([
                        html.Div(id="orig_picture_header" , children=["Original Picture"]),

                        html.Img(id="orig_picture", style={"width" : "29vh"}, className="img-thumbnail"),
                        ], style={"display" : "flex", "flex-direction" : "column", "align-items": "center"}),
                        dbc.Col([
                        html.Div(id="pred_picture_header" , children=["Predicted Segmentation"]),
                        
                        html.Img(id="pred_picture", style={"width" : "29vh"}, className="img-thumbnail"),
                        ], style={"display" : "flex", "flex-direction" : "column", "align-items": "center"}),
                    



                    
                        
                    ], style={"display" : "flex", "flex-direction" : "row", "align-items": "center"} ),
                ], style={"marginTop" : "2vh"})
                ], width = {"size" : 8}),

                dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="hyperparameter_header" , children=["Hyperparameter"]),
                                html.Div(id="bs_text" , children=["Batchsize"]),
                                html.Div(id="lf_text" , children=["Lossfunction"]),
                                html.Div(id="opt_text" , children=["Optimizer"]),
                                html.Div(id="tf_text" , children=["Topologyfactor"]),
                                html.Div(id="ki_text" , children=["Kernelinitialitzer"]),
                            ], style={"justify" : "space-between"}),
                            dbc.Col([
                                html.Div(id="value_header" , children=["Value"]),
                                dcc.Dropdown(id ="bs_dropdown"),
                                dcc.Dropdown(id ="lf_dropdown"),
                                dcc.Dropdown(id ="opt_dropdown"),
                                dcc.Dropdown(id ="tf_dropdown"),
                                dcc.Dropdown(id ="ki_dropdown"),
                            ]),
                            dbc.Col([
                                html.Div(id="checklist_header" , children=["Choose"]),
                                dbc.Checklist(
                                    options=[
                                        {"value": "bs"},
                                        {"value": "lf"},
                                        {"value": "opt"},
                                        {"value": "tf"},
                                        {"value": "ki"},
                                    ],
                                    id="checklist_input",
                                ),
                            ]),
                        ]),
                    ])
                ], style={"marginTop" : "2vh"})
                ], width = {"size" : 4}),
            ], style={"display" : "flex", "flex-direction" : "row", "align-items": "center"}),



        ], width = {"size" : 8}),
 
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Col([
                        dcc.Graph(id="accuracygraph", style={"height" : "42vh"}),
                        dcc.Graph(id="lossgraph", style={"height" : "42vh"})
                    ])
                ],)
            ]),
        ], width = {"size" : 4}),
    ], style={} ),
    dcc.Store(id="current_dataframe"),
    dcc.Store(id="selected_runs"),
    dcc.Store(id="not_selected_runs"),
    dcc.Store(id="change-triggered-by-checkboxes")
], style={ "padding" : "2vh", "min-width" : "inherit"})

], style = {"margin" : 0, "padding-right" : 0, "padding-left" : 0, "min-width" : "100%"})

#updates view of the predicted segmentation
@app.callback([Output(component_id="pred_picture", component_property="src")],
[Input(component_id="similaritygraph",component_property="selectedData"),
Input(component_id="selected_picture_id", component_property="value"),
State(component_id="current_dataframe", component_property="data")],
)

def update_predicted_picture(selectedData, pic_id, data):
    print("callback 'update_predicted_picture' triggered")
    if selectedData != None:
        img_path = ""
        run_id = ""
        x = selectedData["points"][0]["x"]
        y = selectedData["points"][0]["y"]
        df = pd.read_json(data, orient= "index")
        run_id_df = df["run_id"].loc[(df["x"] == x) & (df["y"] == y)]
        run_id_list = run_id_df.to_list()
        if run_id_list:
            run_id = run_id_list[0]
            img_path = "/assets/images/resultsrun3/" + run_id + "/" + pic_id + "{0}.png".format("_predict")
        else:
            img_path = "https://via.placeholder.com/250?text=Select+a+run"

        return [img_path]
    else: 

        img_path = "https://via.placeholder.com/250?text=Select+a+run"

        return [img_path]


#updates view of the original picture
@app.callback(
    [Output(component_id="orig_picture", component_property="src")],
    [Input(component_id="selected_picture_id", component_property="value")],
)

def update_original_picture(pic_id):
    img_path = "/assets/images/originals/" + pic_id + ".png"
    return [img_path]


#updates hyperparameter-table
@app.callback([
    Output(component_id="bs_dropdown", component_property="value"),
    Output(component_id="lf_dropdown", component_property="value"),
    Output(component_id="opt_dropdown", component_property="value"),
    Output(component_id="tf_dropdown", component_property="value"),
    Output(component_id="ki_dropdown", component_property="value"),
    Output(component_id="checklist_input", component_property="value")
],[
    Input(component_id="selected_runs", component_property="data"),
    State(component_id="not_selected_runs", component_property="data"),
    State(component_id="change-triggered-by-checkboxes", component_property="data")
], prevent_initial_call=True)

def update_table(selected_runs_json, not_selected_runs_json, triggered_by_checkboxes):
    if(triggered_by_checkboxes):
        print("callback 'update_table' triggered, BUT by checkboxes, aborting")
        return dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update,dash.no_update

    print("callback 'update_table' triggered by a change in 'selected_runs'")
    triggering_component = callback_context.triggered[0]['prop_id'].split('.')[0]

    #extract all unique hyperparameter options
    bs_value, lf_value, opt_value, tf_value, ki_value = "","","","",""

    if triggering_component != "":

        #extract current run-ids
        selected_run_ids, not_selected_run_ids = extract_current_ids(selected_runs_json,not_selected_runs_json)
        if len(selected_run_ids) == 1:
            selected_parameters = selected_run_ids[0].split("-")
            bs_value = selected_parameters[0][2:]
            lf_value = selected_parameters[1][2:]
            opt_value = selected_parameters[2][3:]
            tf_value = selected_parameters[3][2:]
            print("tf_value in update_table:", tf_value)
            ki_value = selected_parameters[4][2:]

    return bs_value, lf_value, opt_value, tf_value, ki_value, []

#initializes the options for the hyperparameter table
@app.callback([
    Output(component_id="bs_dropdown", component_property="options"),
    Output(component_id="lf_dropdown", component_property="options"),
    Output(component_id="opt_dropdown", component_property="options"),
    Output(component_id="tf_dropdown", component_property="options"),
    Output(component_id="ki_dropdown", component_property="options")
],[
    Input(component_id="current_dataframe", component_property="data")
], prevent_initial_call=True
)

def update_table_options(data):
    print("callback 'update_table_options' triggered by a change in 'current_dataframe'")
    current_df = pd.read_json(data, orient= "index")
    #get all unique options of the different hyperparameters
    bs_options = current_df["batchsize"].unique()
    lf_options = current_df["lossfunction"].unique()
    opt_options = current_df["optimizer"].unique()
    tf_options = current_df["topologyfactor"].unique().tolist()
    for i in range(len(tf_options)):
        tf_options[i] = str(tf_options[i])
    # turn all tf-options into a strings, because of errors when passing floats to the dcc.Dropdown
    ki_options = current_df["kernelinitializer"].unique()
    return bs_options,lf_options,opt_options,tf_options,ki_options



#updates the current dataframe
@app.callback([
    Output(component_id="current_dataframe", component_property="data")
],[
    Input(component_id="selected_picture_id", component_property="value"),
    Input(component_id="selected_dimension", component_property="value"),
])

def update_current_dataframe(selected_picture_id, selected_dimension):
    #simple read and return
    data = pd.read_json("assets/data/embeddata/{0}/{1}.json".format(selected_dimension.lower(),selected_picture_id), orient="index")
    return [data.to_json(orient = "index")]



#updates similaritygraph 
@app.callback([
    Output(component_id="similaritygraph", component_property="figure"),
    ],[
    Input(component_id="selected_picture_id", component_property="value"),
    Input(component_id="selected_dimension", component_property="value"),
    Input(component_id="selected_runs", component_property="data"),
    Input(component_id="not_selected_runs", component_property="data"),
    State(component_id="similaritygraph", component_property="figure")
])

def update_graph(slctd_pic_id, slctd_dim, selected_runs_json, not_selected_runs_json, similarity_fig):

    data = pd.read_json("assets/data/embeddata/{0}/{1}.json".format(slctd_dim.lower(),slctd_pic_id), orient="index")
    triggering_component = callback_context.triggered[0]['prop_id'].split('.')[0]
    print("callback 'update_graph' triggered by {}".format(triggering_component if triggering_component != "" else "initial callback"))

    #change similaritygraph from 2d to 3d or vice versa
    if slctd_dim == "3D":
        similarity_fig = px.scatter_3d(data,x="x",y="y",z="z",color="lossfunction", size="batchsize", symbol="optimizer", custom_data= ["run_id"])
        similarity_fig.update_layout(clickmode='event+select')
    else:
        similarity_fig = px.scatter(data,x="x",y="y",color="lossfunction", size="batchsize", symbol="optimizer", custom_data= ["run_id"])
        similarity_fig.update_layout(clickmode='event+select')

    #if callback is called on page-load, change nothing
    if triggering_component == "":
        return [similarity_fig]
    else:
        #extract current run-ids
    
        selected_run_ids, not_selected_run_ids = extract_current_ids(selected_runs_json,not_selected_runs_json)

        selected_counter = 0
        not_selected_counter = 0
        template_array = np.zeros(len(similarity_fig["data"][0]["customdata"]))
        
        for j in range (len(similarity_fig["data"])):
            # go through selected runs
            
            opacity_array = np.copy(template_array)
            for i in range(len(selected_run_ids)):
                
                for k in range(len(similarity_fig["data"][j]["customdata"])):
                    if similarity_fig["data"][j]["customdata"][k][0] == selected_run_ids[i]:
                        selected_counter += 1
                        np.put(opacity_array, k, 0.8)
                    else:
                        np.put(opacity_array, k, 1)

            # go trough not selected runs
            for i in range(len(not_selected_run_ids)):
                for k in range(len(similarity_fig["data"][j]["customdata"])):
                    if similarity_fig["data"][j]["customdata"][k][0] == not_selected_run_ids[i]:
                        np.put(opacity_array, k, 0.1)
                        not_selected_counter += 1
            similarity_fig["data"][j]["marker"]["opacity"] = opacity_array
        print("sim_fig found {0} selected runs, input was {1} runs".format(selected_counter, len(selected_run_ids)))
        print("sim_fig found {0} not selected runs, input was {1} runs".format(not_selected_counter, len(not_selected_run_ids)))


        return [similarity_fig]
                    




#updates the extended view for loss and accuracy

@app.callback(
    [Output(component_id="accuracygraph", component_property="figure"),
    Output(component_id="lossgraph", component_property="figure")],
    [Input(component_id="selected_runs", component_property="data"),
    State(component_id="not_selected_runs", component_property="data"),
    State(component_id="accuracygraph", component_property="figure"),
    State(component_id="lossgraph", component_property="figure")]
)

def update_extendedview(selected_runs_json, not_selected_runs_json, acc_figInput, loss_figInput):
    print("callback 'update_extendedview' triggered by a change in 'selected_runs'")
    #if something has been clicked
    if selected_runs_json != None:
        #extract current run-ids
    
        selected_run_ids, not_selected_run_ids = extract_current_ids(selected_runs_json,not_selected_runs_json)
       
        #iterate through both existing graphs, to find which line has been selected, then change its style
        inputs = [acc_figInput, loss_figInput]
        for k in range(len(inputs)): #loop trough both graphs
            for j in range(len(selected_run_ids)): #loop through all selected runs
                for i in range(len(inputs[k]["data"])): #loop trough every line in current graph
                    if inputs[k]["data"][i]["name"] == selected_run_ids[j]:
                        inputs[k]["data"][i]["opacity"] = 0.9
                        inputs[k]["data"][i]["line"] = {"dash" : "solid", "width" : 10}
        
        for k in range(len(inputs)): #loop trough both graphs
            for i in range(len(inputs[k]["data"])): #loop trough every line in current graph
                for j in range(len(not_selected_run_ids)): #loop through all selected runs
                    if inputs[k]["data"][i]["name"] == not_selected_run_ids[j]:
                        inputs[k]["data"][i]["opacity"] = 0.1
                        inputs[k]["data"][i]["line"] = {"dash" : "dot", "width" : 3}
        
            
            
        #set output to graphs with updated styles 
        accuracy_fig = acc_figInput
        loss_fig = loss_figInput
    #case: nothing selected
    else:
        #load data as lists
        accuracydata = pd.read_json("assets/data/losslogs/scalars/acc.json", orient="index")
        lossdata = pd.read_json("assets/data/losslogs/scalars/loss.json", orient="index")
        accuracydata_list = accuracydata.values.tolist()
        lossdata_list = lossdata.values.tolist()
        accuracy_fig = go.Figure()
        loss_fig = go.Figure()
        #x-axis
        step = [1,2,3,4,5,6,7,8,9,10]
        index = 0
        #iterate through both dataframes, which both have a length of 108(run)*10(data-entries per run)=1080
        while index < len(accuracydata_list):
            #get the first 10 data-entries -> one run
            acc_data_list_singlerun = accuracydata_list[index:index +10]
            loss_data_list_singlerun = lossdata_list[index:index +10]
            acc_valuelist = []
            loss_valuelist = []
            #append the values that are supposed to be plotted (acc and loss)
            for i in range(10): 
                acc_valuelist.append(acc_data_list_singlerun[i][3])
                loss_valuelist.append(loss_data_list_singlerun[i][3])
            #add one line per run to the plot
            accuracy_fig.add_scatter(x=step, y=acc_valuelist, name=acc_data_list_singlerun[0][0], showlegend = False, hovertext=acc_data_list_singlerun[0][0])
            loss_fig.add_scatter(x=step, y=loss_valuelist, name=loss_data_list_singlerun[0][0], showlegend = False, hovertext=loss_data_list_singlerun[0][0])
            #to skip to the next run in both dataframes
            index += 10
        accuracy_fig.update_layout( xaxis_title="epoch", yaxis_title = "value", 
            title={
            'text': "Accuracy",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'} )
        loss_fig.update_layout( xaxis_title="epoch", yaxis_title = "value",
            title={
            'text': "Loss",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    return accuracy_fig, loss_fig



#updates the selected runs when one of the components, which control selected runs, changes
@app.callback(
    [Output(component_id="selected_runs", component_property="data"),
    Output(component_id="not_selected_runs", component_property="data"),
    Output(component_id="change-triggered-by-checkboxes", component_property="data")],
    [Input(component_id="similaritygraph",component_property="selectedData"),
    Input(component_id="accuracygraph",component_property="selectedData"),
    Input(component_id="lossgraph",component_property="selectedData"),
    Input(component_id="bs_dropdown", component_property="value"),
    Input(component_id="lf_dropdown", component_property="value"),
    Input(component_id="opt_dropdown", component_property="value"),
    Input(component_id="tf_dropdown", component_property="value"),
    Input(component_id="ki_dropdown", component_property="value"),
    Input(component_id="checklist_input", component_property="value"),
    State(component_id="current_dataframe", component_property="data")], prevent_initial_call=True
)

def update_selected_runs(sim_selectedData, acc_selectedData, loss_selectedData, bs_dropdown_value, lf_dropdown_value, opt_dropdown_value, tf_dropdown_value, ki_dropdown_value, checklist_value, data):
    #current dataframe
    current_df = pd.read_json(data, orient= "index")
    #save component which triggered the callback
    triggering_component = callback_context.triggered[0]['prop_id'].split('.')[0]
    #group dropdowns for readabitily
    dropdowns = {"bs_dropdown", "lf_dropdown", "opt_dropdown","tf_dropdown", "ki_dropdown"}
    dropdown_values = {bs_dropdown_value, lf_dropdown_value, opt_dropdown_value, tf_dropdown_value, ki_dropdown_value}



    #callback triggered by similaritygraph
    if triggering_component == "similaritygraph":
        print("callback 'update_selected_runs' triggered by sim_graph with {} runs".format(len(sim_selectedData["points"])))
        if(sim_selectedData != None):
            selected_x_coordinates = []
            selected_y_coordinates = []
            #save only the coordinates from selectedData
            for i in range(len(sim_selectedData["points"])):
                selected_x_coordinates.append(sim_selectedData["points"][i]["x"])
                selected_y_coordinates.append(sim_selectedData["points"][i]["y"])
            #find and save runs in current dataframe where the coordinates match
            selected_runs_df = current_df.loc[current_df["y"].isin(selected_y_coordinates) & current_df["x"].isin(selected_x_coordinates)]
            #save all runs that have not been located above
            not_selected_runs_df = current_df.loc[~current_df["run_id"].isin(selected_runs_df["run_id"])]

            return selected_runs_df.to_json(orient = "index"), not_selected_runs_df.to_json(orient = "index"), False

        else:
            return None, None, False



    #callback triggered by one of the metric-graphs
    elif triggering_component  == "accuracygraph" or triggering_component  == "lossgraph": 
        log_df = None
        selectedData = []

        #load one of the datasets on the metrics to compare later
        if triggering_component  == "accuracygraph":
            print("callback 'update_selected_runs' triggered by acc_graph with {} runs".format(len(acc_selectedData["points"])))
            log_df = pd.read_json("assets/data/losslogs/scalars/acc.json", orient="index")
            selectedData = acc_selectedData

        if triggering_component  == "lossgraph":
            print("callback 'update_selected_runs' triggered by loss_graph with {} runs".format(len(loss_selectedData["points"])))
            log_df = pd.read_json("assets/data/losslogs/scalars/loss.json", orient="index")
            selectedData = loss_selectedData
    
        selected_x_coordinates = []
        selected_y_coordinates = []
        #save only the coordinates from selectedData
        for i in range(len(selectedData["points"])):
            selected_x_coordinates.append(selectedData["points"][i]["x"])
            selected_y_coordinates.append(selectedData["points"][i]["y"])
        #find the runs that match the coordinates
        selected_runs_df_log_format = log_df.loc[log_df["value"].isin(selected_y_coordinates) & (log_df["step"]+1).isin(selected_x_coordinates)]
        #save only the run-ids
        selected_run_ids = selected_runs_df_log_format["run"].tolist()
        #find those run-ids in the current dataframe and save the complete run
        selected_runs_df = current_df.loc[current_df["run_id"].isin(selected_run_ids)]
        #find all runs that have not been located above
        not_selected_runs_df = current_df.loc[~current_df["run_id"].isin(selected_run_ids)]
        return selected_runs_df.to_json(orient = "index"), not_selected_runs_df.to_json(orient = "index"), False
    
    #callback triggered by one of the dropdowns
    elif(triggering_component in dropdowns and not checklist_value):
        #in case of the dropdowns has no value selected
        if(None in dropdown_values):
            print("callback 'update_selected_runs' triggered by one of the dropdowns, BUT at least one value was 'None'")
            return dash.no_update, dash.no_update
        else:
            print("callback 'update_selected_runs' triggered by one of the dropdowns")
            selected_run_ids = []
            tf_dropdown_value = float(tf_dropdown_value)
            #stitch the run-id back together from the selected values in the dropdowns
            selected_run_ids.append("bs" + str(bs_dropdown_value) + "-lf" + str(lf_dropdown_value) + "-opt" + str(opt_dropdown_value) + "-tf" + str(tf_dropdown_value) + "-ki" + str(ki_dropdown_value))
            #find that run in the current dataframe
            selected_runs_df = current_df.loc[current_df["run_id"].isin(selected_run_ids)]
            #save all runs except the one found
            not_selected_runs_df = current_df.loc[~current_df["run_id"].isin(selected_run_ids)]
            return selected_runs_df.to_json(orient = "index"), not_selected_runs_df.to_json(orient = "index"), False
    #callback triggered by one of the checkboxes or a change in one of the dropdowns
    else:
        #(triggering_component == "checklist_input"):
        print("callback 'update_selected_runs' triggered by one of the checkboxes or a change in one of the dropdowns")
        #dict to match checkbox value to the corresponding dropdown
        hp_switcher = {
            "bs" : bs_dropdown_value,
            "lf" : lf_dropdown_value,
            "opt" : opt_dropdown_value,
            "tf" : tf_dropdown_value,
            "ki" : ki_dropdown_value
        }
        checkbox_without_corresponding_value = False
        checked_values = {}
        #loop through all selected checkboxes to see if one has None-value in its corresponding dropdown
        for i in range(len(checklist_value)):
            dropdown_value = hp_switcher.get(checklist_value[i])
            if(dropdown_value == None):
                print("{0}-checkbox checked although it's corresponding dropdown_value is 'None'".format(checklist_value[i]))
                checkbox_without_corresponding_value = True
            else:
                #add {"hyperparameter" : "value"} to checked_values
                cur_hyperparam = switch_hyperparameter(checklist_value[i])
                if cur_hyperparam == "batchsize":
                    dropdown_value = int(dropdown_value)
                elif cur_hyperparam == "topologyfactor":
                    dropdown_value = float(dropdown_value)
                checked_values[cur_hyperparam] = dropdown_value
        #dont change anything if a None-value is found
        if(checkbox_without_corresponding_value):
            return dash.no_update, dash.no_update, dash.no_update
        else:
            print("{} have been selected by the checkboxes".format(checked_values))
            selected_runs_df = current_df
            #find all rows in the dataframe where all conditions are met
            for hyperparameter,value in checked_values.items():
                selected_runs_df = selected_runs_df.loc[selected_runs_df[hyperparameter] == value]
            #looks complicated, but essentially drops all selected rows in selected_runs_df out of current-df
            not_selected_runs_df = current_df.merge(selected_runs_df, indicator=True, how="left")[lambda x: x._merge=='left_only'].drop('_merge',1)
            return selected_runs_df.to_json(orient = "index"), not_selected_runs_df.to_json(orient = "index"), True
            
            

def extract_current_ids(selected_runs_json, not_selected_runs_json):
    selected_runs_df = pd.read_json(selected_runs_json, orient="index")
    not_selected_runs_df = pd.read_json(not_selected_runs_json, orient="index")
    selected_run_ids = selected_runs_df["run_id"].tolist()
    not_selected_run_ids = not_selected_runs_df["run_id"].tolist()
    return selected_run_ids, not_selected_run_ids

def switch_hyperparameter(shortform):
    switcher = {
        "bs" : "batchsize",
        "lf" : "lossfunction",
        "opt" : "optimizer",
        "tf" : "topologyfactor",
        "ki" : "kernelinitializer"
    }
    return switcher.get(shortform)


if __name__ == '__main__':
    app.run_server(debug=True)