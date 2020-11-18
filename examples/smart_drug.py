import numpy as np
from best.model import Model
from best.utils import get_mean_metrics
from best.plot import plot

_ROPE_WIDTH = 0.5


if __name__ == '__main__':
    drug = (101, 100, 102, 104, 102, 97, 105, 105, 98, 101, 100, 123, 105, 103, 100, 95, 102, 106,
            109, 102, 82, 102, 100, 102, 102, 101, 102, 102, 103, 103, 97, 97, 103, 101, 97, 104,
            96, 103, 124, 101, 101, 100, 101, 101, 104, 100, 101)
    placebo = (99, 101, 100, 101, 102, 100, 97, 101, 104, 101, 102, 102, 100, 105, 88, 101, 100,
               104, 100, 100, 100, 101, 102, 103, 97, 101, 101, 100, 101, 99, 101, 100, 100,
               101, 100, 99, 101, 100, 102, 99, 100, 99, 100, 100, 100, 100, 100)

    drug = np.array(drug)
    placebo = np.array(placebo)

    model = Model(drug, placebo)
    posterior, trace = model.sample(it=1000)

    plot(posterior, trace, rope_width=_ROPE_WIDTH, figure_name='smart_drug.png')

    mode, p_above_0, p_rope = get_mean_metrics(trace, rope_width=_ROPE_WIDTH)

    if p_above_0 > 91 and p_rope < 15:
        print('The drug really works, who would have thought')
