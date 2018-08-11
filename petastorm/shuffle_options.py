class ShuffleOptions(object):
    shuffle_row_groups = None
    shuffle_row_drop_partitions = None

    def __init__(self, shuffle_row_groups=True, shuffle_row_drop_partitions=1):
        """
        Constructor.

        :param shuffle_row_groups: Whether to shuffle row groups (the order in which full row groups are read)
        :param shuffle_row_drop_partitions: This is is a positive integer which determines how many partitions to
            break up a row group into for increased shuffling in exchange for worse performance (extra reads).
            For example if you specify 2 each row group read will drop half of the rows within every row group and
            read the remaining rows in separate reads. It is recommended to keep this number below the regular row
            group size in order to not waste reads which drop all rows.
        """
        if not isinstance(shuffle_row_drop_partitions, int) or not shuffle_row_drop_partitions >= 1:
            raise ValueError('shuffle_row_drop_ratio must be positive integer')
        self.shuffle_row_groups = shuffle_row_groups
        self.shuffle_row_drop_partitions = shuffle_row_drop_partitions