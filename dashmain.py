import pandas as pd
import plotly.express as px 
import plotly.graph_objs as go
import base64
from dash import Dash, dcc, html, Input, Output, State 
import os

app = Dash(__name__)

def create_picture_options():
    result = []
    for i in range(30):
        result.append({"label": str(i), "value" : str(i)})
    return result


app.layout  = html.Div([

    html.H1("similarity plot of unet trainings with different hyperparameters", style={'textAlign': 'center'}),

    
    html.Div(id="dropdown_text", children=["select the id of the picture to be analyzed"]),

    dcc.Dropdown(   
        id="selected_picture_id",
        options= create_picture_options(), 
        multi=False,
        value="0", 
        style={"width" : "40%"}  
    ),

    html.Div(id="radio_text", children=["select the dimension to analyze in"]),

    dcc.RadioItems(
        id="selected_dimension",
        options=[{"label" :"2D", "value": "2D"}, {"label": "3D","value":"3D"}], 
        value="2D"),

    dcc.Graph(
        id="similaritygraph",
        style={"height": 400 }
    ),

    html.Div(id="selected_run_output"),

    html.Div(id="picture"),

    dcc.Store(id="current_dataframe")

])

@app.callback([Output(component_id="picture", component_property="children"),
Output(component_id="selected_run_output", component_property="children")],
[Input(component_id="similaritygraph",component_property="clickData"),
Input(component_id="selected_picture_id", component_property="value"),
State(component_id="current_dataframe", component_property="data")],
)

def update_picture(clickData, pic_id, data):
    if clickData != None:
        x = clickData["points"][0]["x"]
        y = clickData["points"][0]["y"]
        print("x:", x)
        print("y:", y)
        df = pd.read_json(data, orient= "index")
        run_id_df = df["run_id"].loc[(df["x"] == x) & (df["y"] == y)]
        print("dataframe:")
        print(run_id_df)
        run_id = run_id_df.to_list()[0]
        dir_path = "results"
        img_path = dir_path + "/" + run_id + "/" + pic_id + "{0}.png".format("" if run_id == "truth" else "_predict")
        print(img_path)
        img_base64 = base64.b64encode(open(img_path, 'rb').read()).decode('ascii')
        output_string = "Selected run: {}".format(run_id)
        return [html.Img(src = "data:image/png;base64,{}".format(img_base64), style={"width":"10%"})], output_string
    else: 
        return [""], ""

    

#update graph itself
@app.callback(
    [Output(component_id="similaritygraph", component_property="figure"),
    Output(component_id="current_dataframe", component_property="data")],
    [Input(component_id="selected_picture_id", component_property="value"),
    Input(component_id="selected_dimension", component_property="value")]
)

def update_graph(slctd_pic_id, slctd_dim):

    data = pd.read_json("C:/Users/momok/Desktop/Bachelorarbeit/dev/results/jsondata/{0}/{1}.json".format(slctd_dim.lower(),slctd_pic_id), orient="index")

    if slctd_dim == "3D":
        fig = px.scatter_3d(data,x="x",y="y",z="z",color="lossfunction", size="batchsize", symbol="optimizer", text="topologyfactor")
        fig.update_layout(clickmode='event+select')
    else:
        fig = px.scatter(data,x="x",y="y",color="lossfunction", size="batchsize", symbol="optimizer", text="topologyfactor")
        fig.update_layout(clickmode='event+select')

    return fig, data.to_json(orient = "index")



if __name__ == '__main__':
    app.run_server(debug=True)