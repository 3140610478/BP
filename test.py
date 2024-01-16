import typing
import bp
import os
import plotly.graph_objects as go
from plotly.offline import plot


def run_test(name: str,
             bpnn: bp.Network,
             dataset: bp.Dataset,
             fig_loss: go.Figure,
             fig_acc: go.Figure,
             epochs: int,
             eta: float | typing.Callable[[int], float] = 0.01,
             save_bpnn: bool = False,
             print_tmps: bool = False):
    rtrain, rtest = [], []
    for i in range(1, epochs+1):
        dataset.reshuffle()
        train = bpnn.train_batch(dataset.train, True, eta)
        test = bpnn.train_batch(dataset.test, False)
        if print_tmps:
            print(i, train, test)
        if i % 10 == 0:
            rtrain.append(train)
            rtest.append(test)

    if save_bpnn:
        bpnn.save_to(bpnn, name + ".bpnn")
    loss, acc = bpnn.train_batch(dataset.Original_data, False)
    print(name, 'loss:', loss, 'accuracy:', acc)

    fig_loss.add_trace(
        go.Scatter(
            name="train loss in [" + name + "]",
            x=list(range(1, epochs+1, 10)),
            y=[i[0] for i in rtrain],
            mode="markers+lines"
        )
    )
    fig_loss.add_trace(
        go.Scatter(
            name="test loss in [" + name + "]",
            x=list(range(1, epochs + 1, 10)),
            y=[i[0] for i in rtest],
            mode="markers+lines"
        )
    )
    fig_acc.add_trace(
        go.Scatter(
            name="train accuracy in [" + name + "]",
            x=list(range(1, epochs+1, 10)),
            y=[i[1] for i in rtrain],
            mode="markers+lines"
        )
    )
    fig_acc.add_trace(
        go.Scatter(
            name="test accuracy in [" + name + "]",
            x=list(range(1, epochs + 1, 10)),
            y=[i[1] for i in rtest],
            mode="markers+lines"
        )
    )


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    epochs = 1000
    dataset = bp.Dataset()
    fig_loss, fig_acc = go.Figure(), go.Figure()
    for t in range(3):
        bpnn = bp.Network([
            bp.Input(4),
            bp.Linear(8),
            bp.Sigmoid(),
            bp.Linear(8),
            bp.Sigmoid(),
            bp.Linear(3),
            bp.Softmax()
        ])
        run_test(str(t), bpnn, dataset, fig_loss,
                 fig_acc, epochs, lambda x: 0.01)
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
    plot(fig_loss, filename="loss.html")
    plot(fig_acc, filename="accuracy.html")
    pass
