import sqlite3 as sqlite


TAXI_DIR_PATH = "../data/sjtu/Taxi/raw/"
BUS_DIR_PATH = "../data/sjtu/Bus/raw/"

def store_one_taxi_record(line, table_name, cur):
	"""
	store one line from SJTU taxi gps files
	line: one line from the raw gps file
	"""
	if len(line.strip()) == 0:
		return
	words = line.split(',') # raw gps file is csv

	# insert data
	insert_sql = "INSERT OR IGNORE INTO {} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)".format(table_name)
	cur.execute(insert_sql, words)

def store_one_bus_record(line, table_name, cur):
	"""
	store one line from SJTU bus gps files
	"""
	if len(line.strip()) == 0:
		return
	words = line.split(',') # raw gps file is csv

	# insert data
	insert_sql = "INSERT OR IGNORE INTO {} VALUES (?, ?, ?, ?, ?, ?, ?)".format(table_name)
	cur.execute(insert_sql, (words[0],words[2],words[6],words[7],words[8],words[12],words[14]))

def store_one_taxi_file(file_name, table_name):
	"""
	store one SJTU taxi gps file
	"""
	file_path = TAXI_DIR_PATH + file_name
	store_one_gps_file(file_path, table_name, store_one_taxi_record)

def store_one_bus_file(file_name, table_name):
	"""
	store one SJTU bus gps file
	"""
	file_path = BUS_DIR_PATH + file_name
	store_one_gps_file(file_path, table_name, store_one_bus_record)


def store_one_gps_file(file_path, table_name, store_one_record):
	"""
	bulk insert to database from taxi or bus gps files
	"""
	con = sqlite.connect('{}.db'.format(table_name))
	con.isolation_level = None # setting for bulk insert (required by Python)
	with open (file_path) as file_input:
		count = 0
		cur = con.cursor()
		bulk_insert_size = 100000 # bulk insert to accelerate the insert process; otherwise each insert is a transaction, toooo slow!!!
		for line in file_input:
			count += 1
			if count % bulk_insert_size == 1:
				print 'start transaction', count
				cur.execute('BEGIN')
			store_one_record(line, table_name, cur)
			if count % bulk_insert_size == 0:
				print 'commit transaction', count
				cur.execute('COMMIT') 

		if count % bulk_insert_size != 0:
			print 'commit transaction', count
			cur.execute('COMMIT') # submit the rest data




if __name__ == '__main__':
	### before insert, needs to create a SQLite database with appropriate table

	#file_name = 'sjtu_taxigps20070207.txt'
	#table_name = 'sh_taxi_20070207'
	#store_one_taxi_file(file_name, table_name)

	file_name = 'busgps20070502.txt'
	table_name = 'sh_bus_20070502'
	store_one_bus_file(file_name, table_name)