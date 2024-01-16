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
    for width in [4,8,16]:
        var = [
            bp.Input(4),
            bp.Linear(width),
            bp.Sigmoid(),
            bp.Linear(width),
            bp.Sigmoid(),
            bp.Linear(3),
            bp.Softmax()
        ]
        run_test('width' + str(width), bp.Network(var), dataset, fig_loss, fig_acc, epochs, 0.01)
        
    fig_loss.update_layout(
        title="loss",
        xaxis={"title": "epoch", 'nticks': int(epochs/100)+1,
            'rangemode': 'tozero', 'range': (0, epochs)},
        yaxis={"title": "loss", 'nticks': 26,
            'rangemode': 'tozero', 'range': (0, 2.5)},
        width=1280,
        height=720,
    )
    fig_acc.update_layout(
        title="accuracy",
        xaxis={"title": "epoch", 'nticks': int(epochs/100)+1,
            'rangemode': 'tozero', 'range': (0, epochs)},
        yaxis={"title": "accuracy", 'nticks': 23,
            'rangemode': 'tozero', 'range': (0, 1.1)},
        width=1280,
        height=720,
    )
    if not os.path.exists("width"):
        os.mkdir("width")
    plot(fig_loss, filename="width/loss_backup.html")
    plot(fig_acc, filename="width/accuracy_backup.html")
    pass
