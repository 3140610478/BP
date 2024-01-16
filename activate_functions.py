import bp
from test import run_test
import os
import plotly.graph_objects as go
from plotly.offline import plot


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    epochs = 1000
    dataset = bp.Dataset()
    fig_loss, fig_acc = go.Figure(), go.Figure()
    Sigmoid = bp.Network([
        bp.Input(4),
        bp.Linear(8),
        bp.Sigmoid(),
        bp.Linear(8),
        bp.Sigmoid(),
        bp.Linear(3),
        bp.Softmax()
    ])
    ReLU = bp.Network([
        bp.Input(4),
        bp.Linear(16),
        bp.LayerNorm(),
        bp.ReLU(),
        bp.Linear(16),
        bp.LayerNorm(),
        bp.ReLU(),
        bp.Linear(3),
        bp.Softmax()
    ])
    run_test("ReLU, width = 16", ReLU, dataset, fig_loss, fig_acc, epochs, 0.01, print_tmps = True)
    run_test("Sigmoid, width = 8", Sigmoid, dataset, fig_loss, fig_acc, epochs, 0.01, print_tmps=True)
    fig_loss.update_layout(
        title="loss",
        xaxis={"title": "epoch", 'nticks': int(epochs/100)+1,
               'rangemode': 'tozero', 'range': (0, 1000)},
        yaxis={"title": "loss", 'nticks': 26,
               'rangemode': 'tozero', 'range': (0, 2.5)},
        width=1280,
        height=720,
    )
    fig_acc.update_layout(
        title="accuracy",
        xaxis={"title": "epoch", 'nticks': int(epochs/100)+1,
               'rangemode': 'tozero', 'range': (0, 1000)},
        yaxis={"title": "accuracy", 'nticks': 23,
               'rangemode': 'tozero', 'range': (0, 1.1)},
        width=1280,
        height=720,
    )
    if not os.path.exists("activate_functions"):
        os.mkdir("activate_functions")
    plot(fig_loss, filename="activate_functions/loss.html")
    plot(fig_acc, filename="activate_functions/accuracy.html")
    pass
