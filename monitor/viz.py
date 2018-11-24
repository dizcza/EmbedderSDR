import os
import time
from collections import defaultdict
from typing import Union, List, Iterable

import numpy as np
import visdom

from monitor.batch_timer import timer


class VisdomMighty(visdom.Visdom):
    def __init__(self, env: str = "main"):
        port = int(os.environ.get('VISDOM_PORT', 8097))
        server = os.environ.get('VISDOM_SERVER', 'http://localhost')
        env = env.replace('_', '-')  # visdom things
        super().__init__(env=env, server=server, port=port)
        print(f"Monitor is opened at {self.server}:{self.port}. Choose environment '{self.env}'.")
        self.timer = timer
        self.legends = defaultdict(list)

    def prepare(self):
        self.register_plot(win='Loss', legend=['batch', 'full train'])
        self.register_plot(win='Accuracy', legend=['full train', 'full test'])

    def register_plot(self, win: str, legend: Iterable[str]):
        legend = list(legend)
        self.legends[win] = legend
        if self.win_exists(win):
            return
        nan = np.full(shape=(1, len(legend)), fill_value=np.nan)
        self.line(X=nan, Y=nan, win=win, opts=dict(legend=legend))

    def line_update(self, y: Union[float, List[float]], opts: dict, name=None):
        y = np.array([y])
        size = y.shape[-1]
        if size == 0:
            return
        if y.ndim > 1 and size == 1:
            y = y[0]
        x = np.full_like(y, self.timer.epoch_progress())
        # hack to make window names consistent if the user forgets to specify the title
        win = opts.get('title', str(opts))
        self.line(Y=y, X=x, win=win, opts=opts, update='append' if self.win_exists(win) else None, name=name)
        if name is not None:
            self.update_window_opts(win=win, opts=dict(legend=self.legends[win]))

    def log(self, text: str):
        self.text(f"{time.strftime('%Y-%b-%d %H:%M')} {text}", win='log', append=self.win_exists(win='log'))
