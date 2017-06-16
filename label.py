import shutil
import os

cmd = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(cmd, 'solution_stg1_release.csv')) as f:
	lines = f.readlines()
	for line in lines:
		fname, label = line.split(',')
		label = label.replace('\n', '')
		ori_path = os.path.join(cmd, 'new', fname)
		new_path = os.path.join(cmd, 'new', 'NType_'+label, fname)
		shutil.move(ori_path, new_path)