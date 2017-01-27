from os. path import dirname, join
from pytest import mark

import matplotlib.pyplot as plt

baseline_dir = join(dirname(__file__), 'test_plot_images')

# Here is an example matplotlib test
#
# The test function must return some object with a savefig method,
# such as a matplotlib figure.
#
# The image created by calling this method is compared with a baseline
# image, if pytest is run with the --mpl option.
#
# The baseline image can be created by running pytest with the
# --mpl-generate-path=<directory> option, where <directiory> specifies where the image will be written.
#
# In order for matplotlib not to crash in headless environments (such
# as Travis) the matplotlib backend must be set to Agg. At the moment
# this is done by hand in the test, but pytest-mpl has recently
# acquired support for this.
@mark.mpl_image_compare(baseline_dir = baseline_dir)
def test_mpl_test_proof_of_concept():
    plt.switch_backend('Agg')
    fig = plt.figure()
    plt.plot([x*x for x in range(10)])
    return fig
