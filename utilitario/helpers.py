
import numpy as np
import os, csv

def get_label_info(csv_path):
    """
    Retorna o nome das classes e seus valores para o dataset.
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
        # print(class_dict)
    return class_names, label_values


def one_hot_it(label, label_values):
    """
    Converte imagem no formato one-hot
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)


    return semantic_map


def reverse_one_hot(image):
    """
    Reverte formato one-hot cria vetor com profundidade 1 com base no maior valor de pixel
    """

    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Converte formato one-hot revertido em imagem segmentada

    """

    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x
