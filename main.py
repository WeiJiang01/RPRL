import yaml
import argparse
from utils.tools import *

if __name__ == '__main__':


    dataset = 'DRWeibo' # Weibo, twibot-22, MGTAB, christianity, twitter

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_filename', default='configs/RPRL/' + dataset +'.yaml', 
                    type=str, help='the configuration to use')
    args = parser.parse_args()

    print(f'Starting experiment with configurations in {args.config_filename}...')
    
    configs = yaml.load(
        open(args.config_filename), 
        Loader=yaml.FullLoader
    )
    args = argparse.Namespace(**configs)

    exec('from supervisor import ' + args.model_name + '_supervisor')
    exec('scores =' + args.model_name + '_supervisor(args)', globals())

                