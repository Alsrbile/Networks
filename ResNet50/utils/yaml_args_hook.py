import os
import yaml
import parser

def configs(config_file, parser):
    
    with open(config_file, encoding="UTF-8") as f:
        args = yaml.safe_load(f) # YAML 데이터를 python 사전형 데이터로 변환함
        
        for k,v in args.items():
            print(k,v)
            parser.add_argument(f"--{k}", default=v, type=type(v))
            
    args = parser.parse_args()
        
    return args