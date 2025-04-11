import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Output, Input, State, callback_context
import uuid

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initial hierarchy data
hierarchy_data = [{"id": str(uuid.uuid4()), "name": "System", "children": [
    {"id": str(uuid.uuid4()), "name": "Subsystem 1", "children": []},
    {"id": str(uuid.uuid4()), "name": "Subsystem 2", "children": []}
]}]

# Function to generate hierarchy layout
def generate_hierarchy(elements):
    if not elements:
        return None
    return html.Ul([
        html.Li([
            html.Span(element["name"], id={'type': 'element-click', 'index': element['id']}, className='clickable-element'),
            dbc.Button("+", id={'type': 'add-element', 'index': element['id']}, size='sm', className='ml-2', color='secondary'),
            generate_hierarchy(element["children"])
        ]) for element in elements
    ])

# Define the layout
app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(
            dbc.Container([
                html.H3("Input Data", className='text-center'),
                dbc.Tabs([
                    dbc.Tab(
                        dbc.Container([
                            html.H4("Product Settings", className='text-center'),
                            html.Div(id='hierarchy-display', children=generate_hierarchy(hierarchy_data)),
                            dbc.Modal([
                                dbc.ModalHeader("Edit Element"),
                                dbc.ModalBody("Placeholder for element settings"),
                                dbc.ModalFooter([
                                    dbc.Button("Close", id="close-modal-btn", color="secondary"),
                                ]),
                            ], id="element-modal", is_open=False),
                        ], fluid=True), label="Product"
                    ),
                    dbc.Tab(
                        dbc.Container([
                            html.H4("Tool Settings", className='text-center'),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Tool Parameter", className='text-center'),
                                    dcc.Input(id='tool-param', type='text', value="Tool1", className='text-center'),
                                ], width=4, className='text-center'),
                            ], justify='center'),
                        ], fluid=True), label="Tools"
                    ),
                    dbc.Tab(
                        dbc.Container([
                            html.H4("Team Settings", className='text-center'),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Team Size", className='text-center'),
                                    dcc.Input(id='team-size', type='number', value=5, className='text-center'),
                                ], width=4, className='text-center'),
                            ], justify='center'),
                        ], fluid=True), label="Teams"
                    ),
                    dbc.Tab(
                        dbc.Container([
                            html.H4("Other Settings", className='text-center'),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Custom Setting", className='text-center'),
                                    dcc.Input(id='custom-setting', type='text', value="Default", className='text-center'),
                                ], width=4, className='text-center'),
                            ], justify='center'),
                        ], fluid=True), label="Other Settings"
                    ),
                ]),
            ], fluid=True, className='text-center'), label="Input Data"
        ),
        dbc.Tab(
            dbc.Container([
                html.H3("Single Run Results", className='text-center'),
                dcc.Graph(id='simulation-graph-1'),
            ], fluid=True, className='text-center'), label="Single Run Results"
        ),
        dbc.Tab(
            dbc.Container([
                html.H3("Monte Carlo Results", className='text-center'),
                dcc.Graph(id='simulation-graph-2'),
            ], fluid=True, className='text-center'), label="Monte Carlo Results"
        ),
    ])
], fluid=True, className='text-center')

# Callback to handle modal popup for element details
@app.callback(
    Output("element-modal", "is_open"),
    Input({'type': 'element-click', 'index': dash.ALL}, "n_clicks"),
    State("element-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_element_modal(n_clicks, is_open):
    return not is_open

# Callback to add a new element dynamically
@app.callback(
    Output("hierarchy-display", "children"),
    Input({'type': 'add-element', 'index': dash.ALL}, "n_clicks"),
    prevent_initial_call=True
)
def add_new_element(n_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    clicked_id = ctx.triggered[0]["prop_id"].split(".")[0]
    for element in hierarchy_data:
        if element["id"] == clicked_id:
            element["children"].append({"id": str(uuid.uuid4()), "name": "New Element", "children": []})
            break
    return generate_hierarchy(hierarchy_data)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
