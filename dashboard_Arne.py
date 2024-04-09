"""Minimal dynamic dash app example.

Click on a button, and draw a new plotly-resampler graph of two noisy sinusoids.
This example uses pattern-matching callbacks to update dynamically constructed graphs.
The plotly-resampler graphs themselves are cached on the server side.

The main difference between this example and 02_minimal_cache.py is that here, we want
to cache using a dcc.Store that is not yet available on the client side. As a result we
split up our logic into two callbacks: (1) the callback used to construct the necessary
components and send them to the client-side, and (2) the callback used to construct the
actual plotly-resampler graph and cache it on the server side. These two callbacks are
chained together using the dcc.Interval component.

"""
from data import get_doocs_properties, load_parquet_data
from pathlib import Path
from datetime import datetime

from typing import List

import plotly.graph_objects as go
from dash import MATCH, Input, Output, State, dcc, html, no_update
from dash_extensions.enrich import (
    DashProxy,
    Serverside,
    ServersideOutputTransform,
    Trigger,
    TriggerTransform,
)
from trace_updater import TraceUpdater

from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB
import pandas as pd


# Data that will be used for the plotly-resampler figures
online_data = {}

doocs_properties = get_doocs_properties(Path("C:/Users/pmahad/PycharmProjects/pythonProject/Database"))


# --------------------------------------Globals ---------------------------------------
app = DashProxy(__name__, transforms=[ServersideOutputTransform(), TriggerTransform()])


app.layout = html.Div(
    [
        html.Div(children=[
        dcc.Dropdown(doocs_properties, id="property-selecter", multi=True),
            html.Button("load data and plot", id="load-plot")
        ]),

        html.Div(id="container", children=[]),
    ]
)


# ------------------------------------ DASH logic -------------------------------------
@app.callback(
    Output("container", "children"),
    Input("load-plot", "n_clicks"),
    State("property-selecter", "value"),
    State("container", "children"),
    prevent_initial_call=True,
)
def add_graph_div(_, selected_properties, div_children: List[html.Div] ):
    climate_children = []
    other_children = []
    if selected_properties is None:
        return []
    else:
        props = [Path(prop) for prop in selected_properties]
        loaded_data = load_parquet_data(props, datetime(2023, 10, 15, 17, 30), datetime(2023, 11, 15, 17, 30))
        for key, item in loaded_data.items():
            online_data[doocs_properties[str(key)]] = item

        div_children = []
        #### make the plots here
        for dat in loaded_data:
            uid = doocs_properties[str(dat)]
            new_child = html.Div(
                children=[
                    # The graph and its needed components to serialize and update efficiently
                    # Note: we also add a dcc.Store component, which will be used to link the
                    #       server side cached FigureResampler object
                    dcc.Graph(id={"type": "dynamic-graph", "index": uid}, figure=go.Figure()),
                    dcc.Loading(dcc.Store(id={"type": "store", "index": uid})),
                    TraceUpdater(id={"type": "dynamic-updater", "index": uid}, gdID=f"{uid}"),
                    # This dcc.Interval components makes sure that the `construct_display_graph`
                    # callback is fired once after these components are added to the session
                    # its front-end
                    dcc.Interval(
                        id={"type": "interval", "index": uid}, max_intervals=1, interval=1
                    ),
                ],
            )

            # Check if the property is CLIMATE
            if "CLIMATE" in str(dat):
                climate_children.append(new_child)
            else:
                other_children.append(new_child)

        # Rearrange other children
        rearranged_others = []
        for child in other_children:
            if "LASER.LOCK.XLO/XTIN" in str(child):
                child.style = {'order': 1}
                rearranged_others.append(child)
            elif "LINK.LOCK" in str(child):
                child.style = {'order': 2}
                rearranged_others.append(child)
            elif "LASER.LOCK.XLO/XHEXP" in str(child):
                child.style = {'order': 3}
                rearranged_others.append(child)

        # all graphs in a row for 'System' tab
        other_tab_content = html.Div(
            rearranged_others,
            style={'display': 'flex', 'flexDirection': 'column'}
        )

        # climate graphs in a separate tab
        climate_tab_content = html.Div(
            climate_children,
            style={'display': 'flex', 'flexDirection': 'column'}
        )

        other_tab = dcc.Tab(label='System', children=other_tab_content)
        climate_tab = dcc.Tab(label='Climate', children=climate_tab_content)

        tabs = dcc.Tabs(children=[other_tab, climate_tab])

        return html.Div(tabs, style={'display': 'flex', 'flexDirection': 'column'})


# This method constructs the FigureResampler graph and caches it on the server side
@app.callback(
    Output({"type": "dynamic-graph", "index": MATCH}, "figure"),
    Output({"type": "store", "index": MATCH}, "data"),
    State("load-plot", "n_clicks"),
    State({"type": "dynamic-graph", "index": MATCH}, "id"),
    Trigger({"type": "interval", "index": MATCH}, "n_intervals"),
    prevent_initial_call=True,
)
def construct_display_graph(n_clicks, analysis) -> FigureResampler:
    fig = FigureResampler(
        go.Figure(),
        default_n_shown_samples=2_000,
        default_downsampler=MinMaxLTTB(parallel=True),
    )

    data = online_data[analysis['index']]
    timestamps = pd.to_datetime(data["timestamp"], unit='s')

    sigma = n_clicks * 1e-6
    fig.add_trace(dict(name="new"), hf_x=timestamps, hf_y=data["data"])
    fig.update_layout(title=f"<b>{analysis['index']}</b>", title_x=0.5)

    return fig, Serverside(fig)


@app.callback(
    Output({"type": "dynamic-updater", "index": MATCH}, "updateData"),
    Input({"type": "dynamic-graph", "index": MATCH}, "relayoutData"),
    State({"type": "store", "index": MATCH}, "data"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata: dict, fig: FigureResampler):
    if fig is not None:
        return fig.construct_update_data(relayoutdata)
    return no_update


# --------------------------------- Running the app ---------------------------------
if __name__ == "__main__":
    app.run_server(debug=True, port=9024)