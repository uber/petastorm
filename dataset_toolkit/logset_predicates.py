#
# Uber, Inc. (c) 2017
#
"""
Predicates for dataset-toolkit
"""
from atg.logsets.client.api import LogSetsClient, Realm
from av.catalog.api.service import get_api_key_from_config, Configuration
from dataset_toolkit import PredicateBase
from av.perception.data_splits.splits import Split, get_log_splits_for_all_volume_guids
from av.perception.data_splits import bootstrap


class in_perception_splits(PredicateBase):
    """ Returns a predicate used to filter frames belonging to a common perception split.
        Common perception log split is used by all ML training/evaluation procedures.
    """

    def __init__(self, split_type, predict_split=False):
        """
        :param split_type: av.perception.data_splits.splits.Split Can be Split.train,
                            Split.test or Split.validation
        :param predic_split: should split for unknown logs be infered using standard algorithm
        """
        if not isinstance(split_type, Split):
            raise ValueError('Split_type is expected to be instance of av.perception.data_splits.splits.Split')
        self._split_type = split_type
        split_dict = get_log_splits_for_all_volume_guids()
        self._volume_guid_field = 'volume_guid'
        self._predict_split = predict_split
        self._included_volume_guids = {str(volume_guid) for volume_guid, split in split_dict.iteritems()
                                       if split == split_type}
        if self._predict_split:
            self._excluded_volume_guids = {str(volume_guid) for volume_guid, split in split_dict.iteritems()
                                           if split != split_type}

    def get_fields(self):
        return {self._volume_guid_field}

    def do_include(self, values):
        if self._volume_guid_field not in values.keys():
            raise ValueError('Tested values does not have split key: %s' % self._volume_guid_field)

        if self._predict_split and (values[self._volume_guid_field] not in self._included_volume_guids) \
                and (values[self._volume_guid_field] not in self._excluded_volume_guids):
            # volume_guid is not assigned to particular split category yet
            # try to predict future split assignment here

            predicted_split = bootstrap.assign_log_split_for_volume_guid(values[self._volume_guid_field])
            return (predicted_split == self._split_type)

        else:
            return values[self._volume_guid_field] in self._included_volume_guids


class in_logset(PredicateBase):
    """ Creates a predicate used for filtering datasets based on the catalog log-sets.
        The predicate is used by dataset readers to filter samples read.
    """

    def __init__(self, logset_name):
        """ logset_name: A log set string as defined by av.perception.utils.logsets.get_logset_revision.
            For example: `odtac_4d_train:8`
        """
        self._logset_name = logset_name
        self._predicate_field = 'index_guid'
        client = LogSetsClient(get_api_key_from_config(Configuration()), Realm.production)
        try:
            name, rev_id = logset_name.split(':')
        except:
            raise ValueError('Failed to split logset name (%s). Format should be <name>:<revision>.' % logset_name)
        logset_revision = client.get_revision(name, rev_id, response_format='all')
        self._included_index_guids = {log['guid'] for log in logset_revision['logs']}

    def get_fields(self):
        return {self._predicate_field}

    def do_include(self, values):
        if self._predicate_field not in values.keys():
            raise ValueError('Tested values does not have split key: %s' % self._predicate_field)
        return values[self._predicate_field] in self._included_index_guids
