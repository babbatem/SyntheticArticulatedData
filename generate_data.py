import os
import time
import argparse

from generation.inspect_data import make_animations

def main(args):

	# initialize Generator
	if args.py_bullet:
		from generation.generator_pybullet import SceneGenerator as SceneGeneratorBullet
		scenegen = SceneGeneratorBullet(root_dir=args.dir,
							  debug_flag=args.debug,
							  masked=args.masked)
	else:
		from generation.generator import SceneGenerator
		scenegen = SceneGenerator(root_dir=args.dir,
							  debug_flag=args.debug,
							  masked=args.masked)

	# make root directory
	os.makedirs(args.dir, exist_ok=True)

	# set generator's target directory for train data
	train_dir = os.path.join(args.dir, args.obj)
	print('Generating training data in %s ' % train_dir)
	os.makedirs(train_dir, exist_ok=False)
	scenegen.savedir = train_dir

	# generate train scenes
	scenegen.generate_scenes(args.n, args.obj, mean_flag=args.mean, left_only=args.left_only, cute_flag=args.cute)

	# set generator's target directory for test data
	test_dir = os.path.join(args.dir, args.obj + '-test')
	os.makedirs(test_dir, exist_ok=False)
	print('Generating test data in %s ' % test_dir)
	scenegen.savedir = test_dir

	# generate test scenes
	scenegen.generate_scenes(int(args.n / 5), args.obj)

	# generate visualization for sanity
	if args.debug:
		make_animations(os.path.join(args.dir,args.obj), min(100, args.n * 16), use_color=args.debug)



parser = argparse.ArgumentParser(description="tool for generating articulated object data")
parser.add_argument('--n', type=int, default=int(1),
					help='number of examples to generate')
parser.add_argument('--dir', type=str, default='../microtrain/')
parser.add_argument('--obj', type=str, default='microwave')
parser.add_argument('--masked', action='store_true', default=False, help='remove background of depth images?')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--mean', action='store_true', default=False, help='generate the mean object')
parser.add_argument('--cute', action='store_true', default=False, help='generate nice shots.')
parser.add_argument('--left-only', action='store_true', default=False, help='generate only left-opening cabinets')
parser.add_argument('--py-bullet', action='store_true', default=False, help='render with PyBullet instead of Mujoco')
main(parser.parse_args())
