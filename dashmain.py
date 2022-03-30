
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, dash_table
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
                        dbc.Col([
                        html.Div(id="selected_run_table_header" , children=["Parameters"]),

                        html.Div(id="selected_run_table"),
                        ], style={"display" : "flex", "flex-direction" : "column", "align-items": "center"}),
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
    dcc.Store(id="current_dataframe")
], style={ "padding" : "2vh"})

], style = {"margin" : 0, "padding-right" : 0, "padding-left" : 0, "min-width" : "100%"})

#updates view of the predicted segmentation
@app.callback([Output(component_id="pred_picture", component_property="src"),
Output(component_id="selected_run_table", component_property="children")],
[Input(component_id="similaritygraph",component_property="clickData"),
Input(component_id="selected_picture_id", component_property="value"),
State(component_id="current_dataframe", component_property="data")],
)

def update_predicted_picture(clickData, pic_id, data):
    if clickData != None:
        x = clickData["points"][0]["x"]
        y = clickData["points"][0]["y"]
        df = pd.read_json(data, orient= "index")
        run_id_df = df["run_id"].loc[(df["x"] == x) & (df["y"] == y)]
        run_id = run_id_df.to_list()[0]
        img_path = "/assets/images/resultsrun3/" + run_id + "/" + pic_id + "{0}.png".format("_predict")

        #create table
        split_id = run_id.split("-")
        table_header = [html.Thead(html.Tr([html.Th("Hyperparameter"), html.Th("Value")]))]
        row1 = html.Tr([html.Td("Batchsize"), html.Td(split_id[0][2:])])
        row2 = html.Tr([html.Td("Lossfunction"), html.Td(split_id[1][2:], style={"width" : "18vh"})])
        row3 = html.Tr([html.Td("Optimizer"), html.Td(split_id[2][3:])])
        row4 = html.Tr([html.Td("Topologyfactor"), html.Td(split_id[3][2:])])
        row5 = html.Tr([html.Td("Kernelinitializer"), html.Td(split_id[4][2:])])

        table_body = [html.Tbody([row1, row2, row3, row4, row5])]

        table = dbc.Table(table_header + table_body, bordered=True,class_name="table")


        return img_path, table
    else: 

        img_path = "https://via.placeholder.com/250?text=Select+a+run"

        table_header = [html.Thead(html.Tr([html.Th("Hyperparameter"), html.Th("Value")]))]
        row1 = html.Tr([html.Td("Batchsize"), html.Td("")])
        row2 = html.Tr([html.Td("Lossfunction"), html.Td("", style={"width" : "18vh"})])
        row3 = html.Tr([html.Td("Optimizer"), html.Td("")])
        row4 = html.Tr([html.Td("Topologyfactor"), html.Td("")])
        row5 = html.Tr([html.Td("Kernelinitializer"), html.Td("")])

        table_body = [html.Tbody([row1, row2, row3, row4, row5])]

        table = dbc.Table(table_header + table_body, bordered=True,class_name="table")

        return img_path, table


#updates view of the original picture
@app.callback(
    [Output(component_id="orig_picture", component_property="src")],
    [Input(component_id="selected_picture_id", component_property="value")],
)

def update_original_picture(pic_id):
    img_path = "/assets/images/originals/" + pic_id + ".png"
    return [img_path]

    

#updates similaritygraph 
@app.callback(
    [Output(component_id="similaritygraph", component_property="figure"),
    Output(component_id="current_dataframe", component_property="data")],
    [Input(component_id="selected_picture_id", component_property="value"),
    Input(component_id="selected_dimension", component_property="value")]
)

def update_graph(slctd_pic_id, slctd_dim):

    data = pd.read_json("assets/data/embeddata/{0}/{1}.json".format(slctd_dim.lower(),slctd_pic_id), orient="index")

    if slctd_dim == "3D":
        fig = px.scatter_3d(data,x="x",y="y",z="z",color="lossfunction", size="batchsize", symbol="optimizer", text="topologyfactor")
        fig.update_layout(clickmode='event+select')
    else:
        fig = px.scatter(data,x="x",y="y",color="lossfunction", size="batchsize", symbol="optimizer", text="topologyfactor")
        fig.update_layout(clickmode='event+select')

    return fig, data.to_json(orient = "index")


#updates the extended view for loss and accuracy

@app.callback(
    [Output(component_id="accuracygraph", component_property="figure"),
    Output(component_id="lossgraph", component_property="figure")],
    [Input(component_id="similaritygraph",component_property="clickData"),
    State(component_id="current_dataframe", component_property="data"),
    State(component_id="accuracygraph", component_property="figure"),
    State(component_id="lossgraph", component_property="figure")]
)

def update_extendedview(clickData, data, acc_figInput, loss_figInput):
    #if something has been clicked
    if clickData != None:
        #extract current run-id
        x = clickData["points"][0]["x"]
        y = clickData["points"][0]["y"]
        df = pd.read_json(data, orient= "index")
        run_id_df = df["run_id"].loc[(df["x"] == x) & (df["y"] == y)]
        run_id = run_id_df.to_list()[0]
        #iterate through both existing graphs, to find which line has been selected, then change its style
        for i in range(len(acc_figInput["data"])):
            if acc_figInput["data"][i]["name"] != run_id:
                acc_figInput["data"][i]["opacity"] = 0.2
                acc_figInput["data"][i]["line"] = {"dash" : "dot","width" : 3}
            else: 
                acc_figInput["data"][i]["opacity"] = 1
                acc_figInput["data"][i]["line"] = {"dash" : "solid", "width" : 10}
            if loss_figInput["data"][i]["name"] != run_id:
                loss_figInput["data"][i]["opacity"] = 0.2
                loss_figInput["data"][i]["line"] = {"dash" : "dot","width" : 3}
            else:
                loss_figInput["data"][i]["opacity"] = 1
                loss_figInput["data"][i]["line"] = {"dash" : "solid", "width" : 10}    
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






if __name__ == '__main__':
    app.run_server(debug=True)