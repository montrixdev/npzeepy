import os, glob, json
from types import ModuleType
import numpy as np
import inspect
import hashlib
import shutil, zipfile

WORKSPACE_KEY = 'workspaces'
WORKSPACE_EXTENSION = '.nzws'
SETTING_PATH = os.path.join(os.getenv('APPDATA'), 'npzee', 'workspace')

def to_python_directory_path(path: str):
    return path.replace('\\', '/')

def to_window_directory_path(path: str):
    return path.replace('/', '\\')

def regist_workspace(ws_name: str, ws_path: str):
    if not os.path.exists(SETTING_PATH):
        raise Exception('npzee install from ms-store is required - ')

    ws_file = os.path.join(SETTING_PATH, ws_name)
    with open(ws_file, 'w+') as f:
        f.write(ws_path.replace('/', '\\'))


def get_workspace(ws_name: str):
    ws_file = os.path.join(SETTING_PATH, ws_name)

    if not os.path.exists(ws_file):
        raise Exception('workspace does not exist - {0}'.format(ws_name))

    with open(ws_file, 'r') as f:
        ws_path = f.read()
        return Workspace(ws_path)


def workspaces_list() -> list:
    return [f for f in os.listdir(SETTING_PATH)]


def workspaces_dict() -> dict:
    d = dict()

    for f in os.listdir(SETTING_PATH):
        with open(os.path.join(SETTING_PATH, f), 'r') as w_f:
            d[f] = w_f.read()

    return d


def support_varialbe_types():
    return (np.ndarray, dict, list, tuple, int, float, str)


def get_current_variables(stack=1):
    d = dict()

    ns = inspect.stack()[stack][0].f_locals
    for k, v in ns.items():
        if not k.startswith('__') and not callable(v) and not isinstance(v, ModuleType) and isinstance(v, support_varialbe_types()):
            d[k] = v

    return d

def numpy_array_hash(arr):
    data_hashcode = hashlib.sha256(np.ascontiguousarray(arr)).hexdigest()
    hash_bytes = str.encode(data_hashcode + str(arr.shape) + str(arr.dtype))
    hashcode = hashlib.sha256(hash_bytes).hexdigest()

    return hashcode


class Workspace:
    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            raise Exception('path does not exist - {0}'.format(path))
        self.path = path

    def _set_variable_value(self, name, v, ext) -> None:
        res_file_path = os.path.join(self.path, name + '.' + ext)
        with open(res_file_path, 'w') as f:
            f.write(str(v))

    def _set_variable_list(self, name, l: list, ext) -> None:
        res_l = dict()
        res_l['list'] = l

        self._set_variable_dict(name, res_l, ext)

    def _set_variable_dict(self, name, d: dict, ext) -> None:
        if d is None:
            raise Exception('dict is empty')

        res_d = dict()
        data_info = dict()

        data_dir_name = name + '.data'
        data_dir_path = os.path.join(self.path, data_dir_name)

        if not os.path.exists(data_dir_path):
            os.makedirs(data_dir_path)

        def get_ndarray(v):
            key_hashcode = hashlib.sha256(str.encode('ndarray' + str(len(data_info)))).hexdigest()
            filename = key_hashcode + '.npy'
            numpy_array_meta = {
                'filename': filename,
                'shape': str(v.shape),
                'dtype': str(v.dtype),
                'data_hashcode': numpy_array_hash(v)
            }

            if key_hashcode not in data_info:
                data_info[key_hashcode] = numpy_array_meta
                filepath = os.path.join(data_dir_path, filename)

                np.save(filepath, v)

            return key_hashcode

        def get_item(item):
            if isinstance(item, np.ndarray):
                return get_ndarray(item)
            elif isinstance(item, dict):
                res_d = dict()
                for k, v in item.items():
                    res_d[k] = get_item(v)
                return res_d
            elif isinstance(item, (list, tuple)):
                res_l = []
                for v in item:
                    res_l.append(get_item(v))
                return res_l
            else:
                return item

        res_d = get_item(d)

        res_d['__data__'] = data_info
        res_d['__meta__'] = {
            'workspace_path': to_window_directory_path(self.path),
            'data_dir_name': data_dir_name
            }

        res_file_path = os.path.join(self.path, name + '.' + ext)
        with open(res_file_path, 'w') as f:
            json.dump(res_d, f)

    def _set_variable_numpy_array(self, name, arr: np.ndarray) -> None:
        if arr is None:
            raise Exception('array is empty')

        nm = name
        filepath = os.path.join(self.path, nm)

        np.save(filepath, arr)

    def set_variable(self, name, value) -> None:
        if isinstance(value, np.ndarray):
            self._set_variable_numpy_array(name, value)
        elif isinstance(value, dict):
            self._set_variable_dict(name, value, 'dict')
        elif isinstance(value, (list, tuple)):
            self._set_variable_list(name, value, 'list')
        elif isinstance(value, (int, float)):
            self._set_variable_value(name, value, 'numeric')
        elif isinstance(value, (str)):
            self._set_variable_value(name, value, 'string')
        else:
            raise Exception('unknown type : {0}'.format(type(value)))

    def _build_variable(self, f_name, f_extension):
        data_info = dict()

        def build_item(item):
            # hashcode
            if isinstance(item, str) and len(item) == 64 and item in data_info:
                data_filepath = os.path.join(self.path, f_name + '.data', data_info[item]['filename'])
                return np.load(data_filepath)
            elif isinstance(item, dict):
                res_d = dict()
                for k, v in item.items():
                    res_d[k] = build_item(v)
                return res_d
            elif isinstance(item, list):
                res_l = []
                for v in item:
                    res_l.append(build_item(v))
                return res_l
            else:
                return item

        file_path = os.path.join(self.path, f_name + f_extension)

        if f_extension == '.npy':
            return np.load(file_path)
        elif f_extension == '.dict' or f_extension == '.list':
            with open(file_path, 'r') as f:
                d = json.load(f)

            data_info = dict()
            data_info = d['__data__']
            del d['__data__']
            del d['__meta__']

            return build_item(d)
        elif f_extension == '.string':
            with open(file_path, 'r') as f:
                return f.read()
        elif f_extension == '.numeric':
            with open(file_path, 'r') as f:
                return float(f.read())
        else:
            pass

    def items(self):
        return [os.path.splitext(f) for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]

    def get_variable(self, name):
        for f in os.listdir(self.path):
            if not os.path.isfile(os.path.join(self.path, f)):
                continue
            f_name, f_extension = os.path.splitext(f)
            if name == f_name:
                return self._build_variable(f_name, f_extension)
        return None

    def has_variable(self, name):
        for f in os.listdir(self.path):
            if not os.path.isfile(os.path.join(self.path, f)):
                continue
            f_name, _ = os.path.splitext(f)
            if name == f_name:
                return True

        return False

    def remove_variable(self, name):
        files = glob.glob(self.path + '/' + name + '*')
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)

    def __setitem__(self, key, value):
        return self.set_variable(key, value)

    def __getitem__(self, key):
        return self.get_variable(key)

    def clear(self):
        files = glob.glob(self.path + '/*')
        for f in files:
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)

    def set_current_variables(self):
        for k, v in get_current_variables(stack=2).items():
            self.set_variable(k, v)

    def to_dict(self):
        res = dict()
        for f in os.listdir(self.path):
            if not os.path.isfile(os.path.join(self.path, f)):
                continue
            f_name, f_extension = os.path.splitext(f)
            res[f_name] = self._build_variable(f_name, f_extension)

        return res

    def import_from_file(self, filename: str):
        if not filename.endswith(WORKSPACE_EXTENSION):
            filename = filename + WORKSPACE_EXTENSION

        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(self.path)

    def export_to_file(self, filename: str):
        def zipdir(path, ziph):
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file))

        if filename.endswith(WORKSPACE_EXTENSION):
            filename = filename.replace(WORKSPACE_EXTENSION, '')

        zipf = zipfile.ZipFile('{0}{1}'.format(filename, WORKSPACE_EXTENSION), 'w', zipfile.ZIP_DEFLATED)
        zipdir(self.path, zipf)
        zipf.close()


if __name__ == '__main__':

    ws = get_workspace(ws_name='ws') # add workspace in npzee
    # ws = Workspace('d:/workspace1') # make directory for workspace first

    d = dict()

    ndarray1 = np.random.randint(0, 100, (2,3))
    ndarray2 = np.random.randint(0, 100, (4,5))
    ndarray3 = np.random.randint(0, 100, (6,7))

    d['ndarray1'] = np.random.randint(0, 100, (2,3))
    d['ndarray2'] = np.random.randint(0, 100, (4,5))
    d['ndarray3'] = np.random.randint(0, 100, (6,7))
    d['str'] = 'string_value'
    d['int'] = int(5)
    d['dict'] = {'test1':1, 'test2':2}
    d['array'] = ['test1',2]
    d['array_ndarray'] = [np.random.randint(0,100,(5,3)), np.random.randint(0,100,(6,4))]

    ws['test'] = d
    ws['test1'] = np.random.randint(0,100,(5,3))

    reload_d = ws.get_variable('test')

    ws['reloaded_test'] = reload_d

