import os
import json
import random
import numpy as np


class Load_dataset(object):
    def __init__(self, spec_path, annotations_path):
        self._spec_path = spec_path
        self._annotations_path = annotations_path
        
        self.annotations = self.npy_load(self._annotations_path)


    # #get spectrograms
    # def load_spectrograms(self, path):
    #     paths = py.glob(path, '*.npy')
    #     spectrograms = [np.abs(np.load(x)) for x in paths]
    #     flns = [os.path.basename(x)[:-4] for x in paths]
    #     return spectrograms, flns

    #load json
    def npy_load(self, path):
        latent_code = [np.load(x) for x in path]
        return latent_code

    def loader(self, shuffle=True):
        flns = [x for x in self.annotations]
        paths = [os.path.join(self._spec_path, 
                              x) for x in flns]
        
        get_annotations = [self.annotations[x] for x in flns]
        
        labels = []
        for i in range(len(get_annotations)):
            labels.append(get_annotations[i]['roles'])
        
        if shuffle:
            c = list(zip(paths, labels, flns))
            random.shuffle(c)
            paths, labels, flns = zip(*c)

        
        return list(paths), list(labels), list(flns)



# load_dataset = Load_dataset(args.dataset_path, args.annotations_json)
# paths, labels, flns = load_dataset.loader()




# class Load_dataset(object):
#     def __init__(self, spec_path, annotations_path):
#         self._spec_path = spec_path
#         self._annotations_path = annotations_path


#     # #get spectrograms
#     # def load_spectrograms(self, path):
#     #     paths = py.glob(path, '*.npy')
#     #     spectrograms = [np.abs(np.load(x)) for x in paths]
#     #     flns = [os.path.basename(x)[:-4] for x in paths]
#     #     return spectrograms, flns

#     def flns(self, path):
#         paths = paths = py.glob(path, '*.npy')
#         flns = [os.path.basename(x)[:-4] for x in paths]
#         return paths, flns

#     #load json
#     def json_load(self, path):
#         with open(path) as f:
#             d = json.load(f)
#         return d


#     #get labels
#     def get_labels(self, annotations_json, flns):
#         annotations = self.json_load(annotations_json)
#         get_annotations = [annotations[x] for x in flns]

#         labels = []
#         for i in range(len(get_annotations)):
#             labels.append(get_annotations[i]['roles'])  #only first 5?

#         return labels

#     # #get data
#     # def get_data(self):
#     #     spectrograms, flns = self.load_spectrograms(self._spec_path)
#     #     labels = self.get_labels(self._annotations_path, flns)
#     #     return spectrograms, labels, flns
    
#     def get_data(self):
#         paths, flns = self.flns(self._spec_path)
#         labels = self.get_labels(self._annotations_path, flns)
#         return paths, labels


# class Load_dataset(object):
#     def __init__(self, spec_path, annotations_path):
#         self._spec_path = spec_path
#         self._annotations_path = annotations_path


#     # #get spectrograms
#     # def load_spectrograms(self, path):
#     #     paths = py.glob(path, '*.npy')
#     #     spectrograms = [np.abs(np.load(x)) for x in paths]
#     #     flns = [os.path.basename(x)[:-4] for x in paths]
#     #     return spectrograms, flns

#     def flns(self, path):
#         paths = paths = py.glob(path, '*.npy')
#         flns = [os.path.basename(x)[:-4] for x in paths]
#         return paths, flns

#     #load json
#     def json_load(self, path):
#         with open(path) as f:
#             d = json.load(f)
#         return d


#     #get labels
#     def get_labels(self, annotations_json, flns):
#         annotations = self.json_load(annotations_json)
#         get_annotations = [annotations[x] for x in flns]

#         labels = []
#         for i in range(len(get_annotations)):
#             labels.append(get_annotations[i]['roles'])  #only first 5?

#         return labels

#     # #get data
#     # def get_data(self):
#     #     spectrograms, flns = self.load_spectrograms(self._spec_path)
#     #     labels = self.get_labels(self._annotations_path, flns)
#     #     return spectrograms, labels, flns
    
#     def get_data(self):
#         paths, flns = self.flns(self._spec_path)
#         labels = self.get_labels(self._annotations_path, flns)
#         return paths, labels
    



