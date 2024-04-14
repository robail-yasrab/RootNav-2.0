import os,hashlib,tempfile,sys,shutil,re,json
from glob import glob
import torch
from .hourglass import hg
from .utils import convert_state_dict
from urllib.request import urlopen

try:
    from tqdm import tqdm
except ImportError:
    # fake tqdm if it's not installed
    class tqdm(object):

        def __init__(self, total=None, disable=False):
            self.total = total
            self.disable = disable
            self.n = 0

        def update(self, n):
            if self.disable:
                return

            self.n += n
            if self.total is None:
                sys.stderr.write("\r{0:.1f} bytes".format(self.n))
            else:
                sys.stderr.write("\r{0:.1f}%".format(100 * self.n / float(self.total)))
            sys.stderr.flush()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.disable:
                return

            sys.stderr.write('\n')

def _download_url_to_file(url, dst, progress):
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    f = tempfile.NamedTemporaryFile(delete=False)

    hash_prefix = None
    pattern = r".*\/[0-9A-Za-z_]*-(?P<hashprefix>.*)\.pkl"
    match = re.match(pattern, url)
    if match is not None:
        hash_prefix = match.groupdict()['hashprefix']

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

class ModelLoader():
    @staticmethod
    def iterate_models():
        # Scan model Directory
        model_dir = os.path.dirname(os.path.realpath(__file__))
        files = glob("{0}/*.json".format(model_dir))
        
        for file in files:
            with open(file, 'r') as f:
                model_json = json.loads(f.read())
                yield model_json

    @staticmethod
    def load_json(name=None, uuid=None, verbose=True):
        if verbose and name is None and uuid is None:
            print ("No model name or uuid specified")
            return None

        model_json = None
        for current_json in ModelLoader.iterate_models():
            if current_json['name'] == name:
                model_json = current_json
                break
            elif uuid is not None and current_json['uuid'] == uuid:
                model_json = current_json
                break

        if verbose and model_json is None:
            if name is not None:
                print ("Model {0} not found".format(name))
            elif uuid is not None:
                print ("Model with uuid {{{0}}} not found".format(uuid))

        return model_json

    @staticmethod
    def list_models(verbose = False):
        model_list = []
        for model_json in ModelLoader.iterate_models():
                if not verbose:
                    model_list.append(model_json['name'])
                else:
                    model_list.append((model_json['name'], model_json['description']))

        return model_list

    @staticmethod
    def model_info(name=None, uuid=None):
        model_json = ModelLoader.load_json(name=name, uuid=uuid)
        if model_json is None:
            return None

        # Model found, construct information
        model_name = model_json['name']
        model_description = model_json['description']
        model_uuid = model_json['uuid']
        model_history = model_json['history']

        multi_plant = model_json['configuration']['multi-plant']

        training_history = []

        # Extract training and training history
        parent_uuid = model_history['model']['parent-model']
        if parent_uuid is not None and parent_uuid != "":
            training_history.append('[{0}]'.format(name))
            while parent_uuid is not None and parent_uuid != "":
                parent_model = ModelLoader.load_json(uuid=parent_uuid, verbose=False)
                training_history.append(parent_model['name'])
                parent_uuid = parent_model['history']['model']['parent-model']

            # Format training history
            col_width = max((len(c) for c in training_history))
            format_code = ["{1:^{0}}".format(col_width, t) for t in training_history]
            symbols = ["{1:^{0}}".format(col_width, "^")] * len(format_code)
            training_history_formatstr = [x for t in zip(format_code, symbols) for x in t]
            del training_history_formatstr[-1]

        # Trained by
        trained_by = model_history['model']['trained-by']
        if isinstance(trained_by, list):
            # Select primary trainer
            trained_by = trained_by[0]

        trained_by_formatstr = "{0} <{1}>, {2}".format(trained_by['fullname'],
                                                       trained_by['email'],
                                                       trained_by['affiliation'])

        # Dataset information
        dataset = model_history['dataset']
        owner = dataset['owner']
        if isinstance(owner, list):
            # Select primary owner
            owner = owner[0]

        dataset_owner_formatstr = "{0} <{1}>, {2}".format(owner['fullname'],
                                                          owner['email'],
                                                          owner['affiliation'])

        dataset_url = dataset['url']

        return [("Model", model_name),
                ("Description", model_description),
                ("Multi plant", "Yes" if multi_plant else "No"),
                ("Trained by", trained_by_formatstr),
                ("Parent Model", "Trained from scratch" if len(training_history) == 0 else "Transferred from {0}".format(training_history[-1])),
                ("Training history", training_history_formatstr if len(training_history) > 0 else []),
                ("Dataset owner", dataset_owner_formatstr),
                ("Dataset URL", dataset_url),
                ("UUID", model_uuid)]

        return output_dict

    @staticmethod
    def get_model(name, gpu=True):
        model_dir = os.path.dirname(os.path.realpath(__file__))
        model_json = ModelLoader.load_json(name=name)
        supported_archs = ['hg']

        if model_json is None:
            raise (Exception("Model not found"))

        selected_arch = model_json['configuration']['network']['architecture']
        if selected_arch not in supported_archs:
            raise (Exception("Model architecture {0} not supported".format(model_json['architecture'])))

        # Load model
        weights_file = "{0}/{1}".format(model_dir, model_json['configuration']['network']['weights'])
        if not os.path.isfile(weights_file):
            # Attempt to download
            sys.stdout.write("Model weight file not found, downloading...")
            sys.stdout.flush()
            _download_url_to_file(model_json['configuration']['network']['url'], weights_file, True)
            print ("Download complete")
            sys.stdout.flush()

        model = None
        if selected_arch == 'hg':
            sys.stdout.write('Loading model...')
            sys.stdout.flush()
            model = hg()
            if gpu:
                state = convert_state_dict(torch.load(weights_file)['model_state'])
            else:
                state = convert_state_dict(torch.load(weights_file, map_location='cpu')['model_state'])
            model.load_state_dict(state)
            model.eval()
            model_json['model'] = model
            print ("Done")

        return model_json
