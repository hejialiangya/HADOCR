import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..', '..')))

from engine import Config
from utility import ArgsParser
import download.utils
from torchvision.datasets.utils import extract_archive

def main(cfg):
    urls, filename_paths, check_validity = download.utils.get_dataset_info(cfg)
    for url, filename_path in zip(urls, filename_paths):
        print(f"Downloading {filename_path} from {url} . . .")
        download.utils.urlretrieve(url=url, filename=filename_path, check_validity=check_validity)
        if not filename_path.endswith(".mdb"):
            extract_archive(from_path=filename_path, to_path=cfg["root"], remove_finished=True)

    print("Downloads finished!")

if __name__ == "__main__":
    FLAGS = ArgsParser().parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    main(cfg.cfg)
