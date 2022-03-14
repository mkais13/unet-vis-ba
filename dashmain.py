import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output  

app = Dash(__name__)

data = pd.read_csv("C:/Users/momok/Desktop/Bachelorarbeit/dev/results/csvdata/3d/25.csv")

fig = px.scatter_3d(data,x="x",y="y",z="z",color="lossfunction", size="batchsize", symbol="optimizer", text="topologyfactor")

app.layout  = html.Div([

    html.H1("similarity plot of unet trainings with different hyperparameters", style={'text-align': 'center'}),

    dcc.Graph(
        id="testgraph",
        figure = fig
        ),

 

])





if __name__ == '__main__':
    app.run_server(debug=True)