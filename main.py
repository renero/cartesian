from cartesian import Cartesian
import yaml
from os.path import join, isdir


# if __name__ == "__main__":
# Open configuration file
with open("main.yml", 'r') as ymlfile:
    params = yaml.load(ymlfile)

for uc_name in params['uc_names']:
    input_folder = join(params['uc_path'], uc_name)
    if isdir(input_folder):
        print('Entering: {}'.format(input_folder))
        combiner = Cartesian(input_folder)
        utterances = combiner.product(save=True)
    print('done.')
