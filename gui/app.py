# Imports
import dash
from dash import dcc, no_update
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format, Scheme
import plotly.graph_objs as go
import threading
import pandas as pd
import warnings

# Relative imports
from database.connector import MongoDBConnector
from analytics import DashTable, BackTest
from utils import get_ticker_name

# Turn off warnings (from optimial parameters)
warnings.filterwarnings("ignore")

# Initialize Dash app
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Equity Pairs Trading Tool"
app.update_title = None
server = app.server

# Initialize MongoDB connector
db_connector = MongoDBConnector()

####################################
# LAYOUT
####################################
app.layout = html.Div(
    [
        # Header
        html.H1("Equity Pairs Trading Tool"),
        html.H3(
            "1. Select the time range, number of pairs and selection criterion."
        ),
        html.H3(
            "2. Press 'Compute pairs'."
        ),
        html.H3(
            "3. Select a pair to backtest a trading strategy."
        ),
        html.H3(
            "4. Experiment with backtesting parameters."
        ),
        html.H5(
            f"Equities included: individual components (and indices) of the S&P 500, NASDAQ 100, and Russell 2000. Total number of equities included: {len(db_connector.tickers_in_db)}"
        ),
        html.H6("Notes: (i) Data is only available from 1 Jan 2020 to 31 Jul 2023."),
        html.H6(
            "(ii) In order to reduce computation time, only pairs with above 0.95 correlation are considered."
        ),
        html.Br(),
        # Date Picker for start date
        html.Label("Start Date:"),
        dcc.DatePickerSingle(id="start-date-picker", date="2022-01-01"),
        # Date Picker for end date
        html.Label("End Date:"),
        dcc.DatePickerSingle(id="end-date-picker", date="2022-12-31"),
        # Number of pairs
        html.Label("Number of Pairs:"),
        dcc.Input(
            id="num_of_pairs",
            type="number",
            value=100,  # Default value
            min=1,
            step=1,
        ),
        html.Br(),
        # Dropdown for criteria selection
        html.Label("Criterion for Pairs Selection:"),
        dcc.Dropdown(
            id="pairs-criteria-dropdown",
            options=[
                {"label": "Correlation", "value": "Correlation"},
                {
                    "label": "Mean Reversion Half Life",
                    "value": "Mean Reversion Half Life",
                },
                {"label": "Cointegration P-Value", "value": "Cointegration P-Value"},
            ],
            value="Cointegration P-Value",  # Default value
            clearable=False,  # Prevents the user from clearing the selection
            style={"width": "50%"},
        ),
        # Break
        html.Br(),
        # Button to compute correlations
        html.Button("Compute pairs", id="submit-button", n_clicks=0),
        # Error message
        html.Div(id="input-error-message"),
        # Interval component to update the display
        dcc.Store(id="process-state"),
        dcc.Interval(
            id="interval-component",
            interval=1e12,  # some massive number, essentially inf
            n_intervals=0,
            max_intervals=1,  # Only run once
        ),
        # Display area for results (tables, graphs, etc.)
        html.Div(
            [
                html.Div(
                    [  # This div will contain the table and be on the left
                        html.Div(id="display-results"),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "vertical-align": "top",
                    },
                ),
                html.Div(
                    [
                        dcc.Graph(id="price-plot"),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "vertical-align": "top",
                    },
                ),
            ],
            style={"display": "flex", "flex-direction": "row"},
        ),
        html.H2("Backtesting Strategy"),
        html.H5(
            "A simple mean reversion strategy is employed, based on z-scores. The z-score is calculated as: (spread - rolling mean of spread) / rolling standard deviation of spread. If the z-score is above the threshold, then a long position in the cheaper stock and a short position in the more expensive stock are taken. If the z-score is below the negative of the threshold, then a short position in the cheaper stock and a long position in the more expensive stock are taken."
        ),
        html.H5(
            "You can adjust the threshold and rolling window below, and press Find Optimal Parameters to see the parameters that maximise the Sharpe Ratio."
        ),
        html.H5(
            "A z-score threshold of 1.0 and a rolling window of 30 days are used by default."
        ),
        html.H6(
            "Notes (i): The risk-free rate has been assumed to be the Effective Fed Funds Rate (EFFR)."
        ),
        html.H6(
            "(ii) 'Find Optimal Parameters' is a rudimentary optimisation method (Bayesian Optimisation), and may not be the true optimal parameters. Optimal parameters may be different every time you compute optimal parameters."
        ),
        # Options for backtesting
        html.Div(
            [
                html.Label("z-score Threshold:"),
                dcc.Input(id="z-threshold", type="number", value=1.0, step=0.01),
                html.Label("Rolling Window:"),
                dcc.Input(id="rolling_window", type="number", value=30, step=0.01),
            ]
        ),
        # Optimise button
        html.Button("Find Optimal Parameters", id="optimize-button", n_clicks=0),
        html.Div(id="optimal-params-display"),
        html.Div(
            [html.Div(id="backtest-table"), dcc.Graph(id="strategy-performance-plot")],
            style={
                "width": "50%",
                "display": "inline-block",
                "vertical-align": "top",
            },
        ),
        # Backtesting graphs
        html.Div(
            [
                dcc.Graph(id="backtest-graph"),
            ],
            style={
                "width": "50%",
                "display": "inline-block",
                "vertical-align": "top",
            },
        ),
    ]
)

####################################
# MAIN TABLE
####################################
# Global variable to store the computed data
done_event = threading.Event()
computed_data = None


def threaded_task(
    start_date,
    end_date,
    num_of_pairs,
    pairs_criteria=0.95,
    selection_criteria="Correlation",
):
    """
    Function to run in the background to compute the data for the table.
    :param start_date: The start date.
    :param end_date: The end date.
    :param num_of_pairs: The number of pairs to return in the table
    :param pairs_criteria: The minimum correlation considered
    :param selection_criteria: The criteria for selecting the pairs.
    """
    global computed_data
    analytics = DashTable(start_date, end_date, num_of_pairs, pairs_criteria)
    # Limit down to only where correlation is above criteria
    analytics.compute_top_correlations()
    # Compute data for the table
    computed_data = analytics.stock_selection(selection_criteria)
    done_event.set()


# CALLBACK TO UPDATE THE TABLE
@app.callback(
    [
        Output("display-results", "children"),
        Output("input-error-message", "children"),
        Output("process-state", "data"),
        Output("interval-component", "interval"),
        Output("interval-component", "max_intervals"),
    ],  # Add this output
    [Input("submit-button", "n_clicks"), Input("interval-component", "n_intervals")],
    [Input("pairs-criteria-dropdown", "value")],
    [
        State("start-date-picker", "date"),
        State("end-date-picker", "date"),
        State("num_of_pairs", "value"),
        State("process-state", "data"),
    ],
)
def update_display(
    n_clicks,
    n_intervals,
    selected_criteria,
    start_date,
    end_date,
    num_of_pairs,
    process_state,
):
    ctx = dash.callback_context
    if not ctx.triggered:
        triggered_id = None
    else:
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # START THE BACKGROUND TASK AND SHOW A SPINNER
    if triggered_id == "submit-button":
        # If the button is clicked, reset the done_event and clear the previously computed data
        done_event.clear()
        global computed_data
        computed_data = None

        # Start the background task and immediately return a spinner
        thread = threading.Thread(
            target=threaded_task,
            args=(start_date, end_date, num_of_pairs, 0.95, selected_criteria),
        )
        thread.start()
        spinner = dbc.Spinner(size="lg", color="primary")
        return (
            spinner,
            "",
            "processing",
            1000,
            1e12,
        )  # Start checking every second and let it run indefinitely

    # WHEN BACKGROUND TASK IS DONE, DISPLAY THE RESULTS
    elif triggered_id == "interval-component" and done_event.is_set():
        data, cols = computed_data
        number_format = Format(precision=3, scheme=Scheme.fixed)
        formatted_cols = [
            {**col, "type": "numeric", "format": number_format} for col in cols
        ]
        table = dash_table.DataTable(
            data=data,
            columns=formatted_cols,
            page_size=10,
            id="table",
            row_selectable="single",
            sort_action="native",
        )
        return (
            table,
            "",
            "done",
            1e12,
            n_intervals,
        )  # Stop the interval component by setting max_intervals to its current value

    # IF NO UPDATE, DO NOTHING
    else:
        return no_update, no_update, no_update, no_update, no_update  # No updates


####################################
# PAIRS GRAPH
####################################
@app.callback(
    Output("price-plot", "figure"),
    [
        Input("table", "selected_rows"),
        Input("start-date-picker", "date"),
        Input("end-date-picker", "date"),
    ],
    [State("table", "data")],
)
def update_graph(selected_rows, start_date, end_date, table_data):
    if not selected_rows:  # If no rows are selected, return an empty figure
        return {}

    # Get the selected pair from the table data
    selected_pair = table_data[selected_rows[0]]["Pair"]
    stock1, stock2 = selected_pair.split(" - ")

    # Get name of the stocks
    stock1_name = get_ticker_name(stock1)
    stock2_name = get_ticker_name(stock2)

    # Fetch the price data for the selected stocks within the date range
    stock_1_price = db_connector.fetch_single_stock_on_time_range(
        stock1, start_date, end_date
    )
    stock_2_price = db_connector.fetch_single_stock_on_time_range(
        stock2, start_date, end_date
    )

    # Create the plot
    fig = {
        "data": [
            {
                "x": stock_1_price.index,
                "y": stock_1_price["close"],
                "type": "line",
                "name": stock1,
            },
            {
                "x": stock_2_price.index,
                "y": stock_2_price["close"],
                "type": "line",
                "name": stock2,
            },
        ],
        "layout": {
            "title": f"Price Data for {selected_pair}<br><sup>{stock1_name} and {stock2_name}</sup>",
            "labels": {"x": "Date", "y": "Price (USD)"},
        },
    }
    return fig


####################################
# BACKTEST TABLE AND SPREAD GRAPH
####################################
# BACKTESTING
@app.callback(
    [Output("backtest-graph", "figure"), Output("backtest-table", "children")],
    [
        Input("table", "selected_rows"),
        Input("start-date-picker", "date"),
        Input("end-date-picker", "date"),
        Input("z-threshold", "value"),
        Input("rolling_window", "value"),
    ],
    [State("table", "data")],
)
def update_backtest_graph(
    selected_rows, start_date, end_date, z_thresh, rolling_window, table_data
):
    if not selected_rows:
        return {}

    # Get the selected pair from the table data
    selected_pair = table_data[selected_rows[0]]["Pair"]
    stock1, stock2 = selected_pair.split(" - ")

    # Run backtest
    backtest = BackTest(stock1, stock2, start_date, end_date)
    backtest.backtest(window=rolling_window, z_thresh=z_thresh)
    signal = backtest.signal
    (
        sharpe_ratio,
        sortino_ratio,
        max_drawdown,
        annualized_return,
        annualized_volatility,
    ) = backtest.calculate_metrics()

    # Create the table
    data_dict = {
        "Metric": [
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Annualized Return",
            "Annualized Volatility",
        ],
        "Value": [
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            annualized_return,
            annualized_volatility,
        ],
    }
    df = pd.DataFrame(data_dict)
    data = df.to_dict("records")

    metrics = {
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
    }

    formatted_metrics = {}
    for key, value in metrics.items():
        if key in ["Max Drawdown", "Annualized Return", "Annualized Volatility"]:
            formatted_metrics[key] = "{:.2%}".format(value)
        else:
            formatted_metrics[key] = "{:.2f}".format(value)

    data_dict = {
        "Metric": list(metrics.keys()),
        "Value": list(formatted_metrics.values()),
    }
    df = pd.DataFrame(data_dict)
    data = df.to_dict("records")

    columns = [{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}]

    table = dash_table.DataTable(data=data, columns=columns)

    # Create the plot
    fig = go.Figure()

    # Add the main line plot for spread
    fig.add_trace(
        go.Scatter(
            x=backtest.signal.index, y=backtest.spread, mode="lines", name="Spread"
        )
    )

    # Add background shading based on signal
    shapes = []
    dates = signal.index.tolist()
    signal = signal.tolist()
    for i, sig in enumerate(signal):
        if i == len(signal) - 1:
            continue

        color = None
        if sig == 1:
            color = "green"
        elif sig == -1:
            color = "red"

        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": dates[i],
                "y0": 0,
                "x1": dates[i + 1],
                "y1": 1,
                "fillcolor": color,
                "opacity": 0.2,
                "line_width": 0,
            }
        )

    fig.update_layout(
        shapes=shapes,
        title=f"Spread for {selected_pair}<br><sup>Green and red regions represent long and short positions in the spread</sup>",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title_font_color="black",
        xaxis=dict(
            title="Date",
            titlefont=dict(color="black"),
            tickfont=dict(color="black"),
            gridcolor="lightgrey",
        ),
        yaxis=dict(
            title="Spread (USD)",
            titlefont=dict(color="black"),
            tickfont=dict(color="black"),
            gridcolor="lightgrey",
        ),
    )
    return fig, table


####################################
# PRINT OUT OPTIMAL PARAMETERS
####################################
@app.callback(
    Output("optimal-params-display", "children"),
    [Input("optimize-button", "n_clicks")],
    [
        State("start-date-picker", "date"),
        State("end-date-picker", "date"),
        State("table", "selected_rows"),
        State("table", "data"),
    ],
)
def find_optimal_params(n_clicks, start_date, end_date, selected_rows, table_data):
    if not selected_rows or n_clicks == 0:
        return dash.no_update

    # Get the selected pair from the table data
    selected_pair = table_data[selected_rows[0]]["Pair"]
    stock1, stock2 = selected_pair.split(" - ")

    backtest_optim = BackTest(stock1, stock2, start_date, end_date)
    optimal_params = backtest_optim.find_optimal_params()

    # Format and display the parameters
    return html.P(
        f"Parameters to optimise the Sharpe Ratio: z-score: {optimal_params['z_thresh']}, rolling window: {optimal_params['window']}"
    )


####################################
# CUMULATATIVE RETURNS PLOT
####################################
@app.callback(
    Output("strategy-performance-plot", "figure"),
    [
        Input("table", "selected_rows"),
        Input("z-threshold", "value"),
        Input("rolling_window", "value"),
    ],
    [
        State("start-date-picker", "date"),
        State("end-date-picker", "date"),
        State("table", "data"),
    ],
)
def update_strategy_performance(
    selected_rows, z_threshold, rolling_window, start_date, end_date, table_data
):
    if not selected_rows:
        return dash.no_update

    # Get the selected pair from the table data
    selected_pair = table_data[selected_rows[0]]["Pair"]
    stock1, stock2 = selected_pair.split(" - ")

    backtest = BackTest(stock1, stock2, start_date, end_date)
    backtest.backtest(window=rolling_window, z_thresh=z_threshold)

    fig = {
        "data": [
            {
                "x": backtest.cumulative_returns.index,
                "y": backtest.cumulative_returns,
                "type": "line",
            },
        ],
        "layout": {
            "title": f"Cumulatative Returns",
            "labels": {"x": "Date", "y": "Returns"},
        },
    }

    return fig


if __name__ == "__main__":
    app.run_server(debug=False)
