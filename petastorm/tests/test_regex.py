# testing ngram regex match 
from petastorm import make_reader
from petastorm.ngram import NGram
from petastorm.tests.test_common import create_test_dataset, TestSchema
from av.perception.datasets.msds.schema import MsdsSchema


def check_regex_match():
	parquet_ds_url = 'file:///home/dimitrov/emorara20'

	fields = ['lidar_point_laser_row', 'labels_*']

	reader = make_reader(dataset_url=parquet_ds_url, schema_fields=fields)

	#print (next(reader))


def ngram_creation():
	parquet_ds_url = 'file:///home/dimitrov/emorara20'

	# ts_field = 'lidar_start_capture_time_sec'

	# fields = {
	# -1 : ['lidar_point_laser_row', 'lidar_start_capture_time_sec'],
	#  0: ['lidar_point_laser_row', 'lidar_start_capture_time_sec'],
	#  1: ['lidar_point_laser_row', 'lidar_start_capture_time_sec'],
	# }

	# fields = {
	# -1 : [MsdsSchema.lidar_point_laser_row, MsdsSchema.lidar_start_capture_time_sec],
	#  0: [MsdsSchema.lidar_point_laser_row, MsdsSchema.lidar_start_capture_time_sec],
	#  1: [MsdsSchema.lidar_point_laser_row, MsdsSchema.lidar_start_capture_time_sec],
	# }

	ts_field_original=MsdsSchema.lidar_start_capture_time_sec


	fields = {
	-1 : ['lidar_start_capture_time_sec', 'labels_*', MsdsSchema.lidar_end_capture_time_sec],
	 0: ['lidar_start_capture_time_sec', 'labels_*', MsdsSchema.lidar_end_capture_time_sec],
	 1: ['lidar_start_capture_time_sec', 'labels_*', MsdsSchema.lidar_end_capture_time_sec],
	}

	ts_field = 'lidar_end_capture_time_sec'

	ngram = NGram(fields=fields,
				  delta_threshold=.5,
				  timestamp_field=ts_field)


	reader = make_reader(dataset_url=parquet_ds_url, schema_fields=ngram)

	print (next(reader))

if __name__ == '__main__':
	#check_regex_match()

	ngram_creation() 