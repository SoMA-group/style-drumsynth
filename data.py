import tf2lib as tl

def make_custom_dataset(audio_paths, batch_size, labels, resize, drop_remainder=True, shuffle=True, repeat=1):

    dataset = tl.disk_image_batch_dataset(audio_paths,
                                          batch_size,
                                          size=resize,
                                          labels=labels,
                                          drop_remainder=drop_remainder,
#                                          map_fn=_map_fn,
                                          shuffle=shuffle,
                                          repeat=repeat)
    audio_shape = (resize, 1)
    len_dataset = len(audio_paths) // batch_size

    return dataset, audio_shape, len_dataset


