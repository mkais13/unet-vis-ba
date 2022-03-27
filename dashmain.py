
import pandas as pd
import plotly.express as px 
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


app.layout = html.Div([
dbc.NavbarSimple(color="primary",brand = "SIMILARITY PLOT OF UNET TRAININGS WITH DIFFERENT HYPERPARAMETERS", dark=True),

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
                                    style={"padding-right" : 29}
                                ),
                            ], className="text-center"),

                        ],align="center"),
                    ], justify= "evenly", align="center"),
                    dcc.Graph(id="similaritygraph"),
                ]),
            ]),
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dcc.Graph(id="extendedview"),
                    ]),
                ]),
            ]),
        ], width = {"size" : 9}),
 
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Col([
                        html.Div(id="orig_picture_header" , children=["Original Picture"], className="text-center"),

                        html.Img(id="orig_picture", style={"width" : "250px", "min-heigth" : "250px"}, className="img-thumbnail"),
            
                        html.Div(id="pred_picture_header" , children=["Predicted Segmentation"], className="text-center"),

                        html.Img(id="pred_picture", style={"width" : "250px", "min-heigth" : "250px"}, className="img-thumbnail"),
                
                        html.Div(id="selected_run_table_header" , children=["Parameters"], className="text-center"),

                        html.Div(id="selected_run_table"),
                    ], style={"display" : "flex", "flex-direction" : "column", "align-items": "center"})
                ])
            ]),
        ], width={"size" : 3}),
    ]),
    dcc.Store(id="current_dataframe")
], style={"max-width" : "95%", "paddingTop" : "10px"})

])

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
        img_path = "/assets/images/resultsrun3/" + run_id + "/" + pic_id + "{0}.png".format("" if run_id == "truth" else "_predict")

        #create table
        split_id = run_id.split("-")
        table_header = [html.Thead(html.Tr([html.Th("Hyperparameter"), html.Th("Value")]))]
        row1 = html.Tr([html.Td("Batchsize"), html.Td(split_id[0][2:])])
        row2 = html.Tr([html.Td("Lossfunction"), html.Td(split_id[1][2:], style={"width" : "170px"})])
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
        row2 = html.Tr([html.Td("Lossfunction"), html.Td("", style={"width" : "170px"})])
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

    data = pd.read_json("assets/data/jsondata/{0}/{1}.json".format(slctd_dim.lower(),slctd_pic_id), orient="index")

    if slctd_dim == "3D":
        fig = px.scatter_3d(data,x="x",y="y",z="z",color="lossfunction", size="batchsize", symbol="optimizer", text="topologyfactor")
        fig.update_layout(clickmode='event+select')
    else:
        fig = px.scatter(data,x="x",y="y",color="lossfunction", size="batchsize", symbol="optimizer", text="topologyfactor")
        fig.update_layout(clickmode='event+select')

    return fig, data.to_json(orient = "index")

@app.callback(
    [Output(component_id="extendedview", component_property="figure")],
    [Input(component_id="similaritygraph",component_property="clickData")]
)

def update_extendedview(clickData):
    if clickData != None:
        x = clickData["points"][0]["x"]
        y = clickData["points"][0]["y"]
    experiment_id = "A1B5RFrYQhGb4eVOBkibUA"
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    print("test")
    print("df:",df)
    fig = px.line(df, x="step", y ="epoch_loss")
    return fig



if __name__ == '__main__':
    app.run_server(debug=True)