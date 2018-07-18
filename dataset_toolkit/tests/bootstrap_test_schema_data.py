#
# Uber, Inc. (c) 2018
#
import getopt
import sys

from dataset_toolkit.tests.test_common import create_test_dataset


# Number of rows in a fake dataset
ROWS_COUNT = 10


def usage_exit(msg=None):
    if msg:
        print(msg)
    print("""\
Usage: {} [options]

Options:
  -h, --help           Show this message
  --output-dir <dir>   Path of directory where to write test data
""".format(sys.argv[0]))
    sys.exit(1)


def make_test_metadata(path):
    """
    Use test_common to make a dataset for the TestSchema .

    If you are updating the unit test data version, be sure to follow
    https://code.int.uberatc.com/w/best_practices/testing/#how-to-store-your-test-d
    and request to copy the data to the build machines.

    :param path: path to store the test dataset
    :return: resulting dataset as a dictionary
    """
    assert len(path) > 0, 'Please supply a nonempty path to store test dataset.'
    return create_test_dataset('file://{}'.format(path), range(ROWS_COUNT))


if __name__ == '__main__':
    try:
        options, args = getopt.getopt(sys.argv[1:], 'ho:', ['--help', 'output-dir='])
        path = None
        for opt, value in options:
            if opt in ('-h', '--help'):
                usage_exit()
            if opt in ('-o', '--output-dir'):
                if len(value) > 0:
                    path = value
        if path is None or len(path) == 0:
            usage_exit('Please supply an output directory.')
        else:
            make_test_metadata(value)
    except getopt.GetoptError as msg:
        usage_exit(msg)
