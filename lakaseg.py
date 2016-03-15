#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division, unicode_literals
import ctypes
import platform
import sys
import os


if platform.system() == "Windows":
    DLL_PATH = "./Release/Lakaseg_lib.dll"
else:
    DLL_PATH = "./liblakaseg.so"

directory_of_this_file = os.path.dirname(os.path.realpath(__file__))
DLL_PATH = os.path.normpath(os.path.join(directory_of_this_file, DLL_PATH))
lakaseg_lib = ctypes.CDLL(DLL_PATH)

is_python3 = (sys.version_info.major == 3)
if is_python3:
    def encode_str(s):
        return bytes(s, 'utf-8')
else:
    def encode_str(s):
        return s


def trainieren(training_data, target_json_file,
               forest_size=7,
               max_tree_depth=3,
               testobject_tries=600,
               window_size=9,
               number_of_threads=0):
    """
    training_data: entweder ein Tupel (trainingsbild.png, labels.png) oder
    eine Liste [(trainingsbild1.png, labels1.png), (trainingsbild2.png,
    labels2.png), ...], wenn mit mehreren Bildern trainiert werden soll

    target_json_file: der Dateiname für die JSON-Datei, in der die gelernten
    Bäume gespeichert werden.
    Muss unter Windows evtl. ein absoluter Pfad sein.

    forest_size: Anzahl der Bäume, die gelernt werden. Je, mehr, desto länger
    dauern Training und Inferenz, aber die Leistung ist i.A. besser

    max_tree_depth: die maximale Tiefe der Entscheidungsbäume. Mit steigender
    Tiefe sollte die Leistung besser und danach schlechter werden, weil die
    Entscheidungsbäume sich dann stärker an die Trainingsbilder anpassen und
    weniger gut auf andere Testbilder generalisieren. Training und Inferenz
    dauern umso länger, je höher der Wert.

    testobject_tries: die Anzahl der Versuche beim Sampeln von Testobjekten in
    den Entscheidungsbäumen. Es gilt das gleiche wie bei max_tree_depth.

    window_size: wie groß das Fenster um ein Pixel herum ist. Die Pixel in
    diesem Fenster werden zur Klassifizierung herangezogen. Muss eine ungerade
    Zahl sein. Die Auswirkungen waren in den Tests nicht groß.

    number_of_threads: mit OpenMP parallelisieren. Der Wert 1 schaltet die
    Parallelisierung aus, > 1 spezifiziert die Anzahl der Threads, 0 lässt
    OpenMP automatisch die Anzahl von Threads wählen.
    """

    if window_size < 1 or window_size % 2 != 1:
        print("Fehler: Die Fenstergröße muss eine ungerade Zahl > 0 sein",
              file=sys.stderr)
        exit(1)
    window_radius = (window_size - 1) // 2

    if isinstance(training_data, tuple):
        training_data = [training_data]

    if not target_json_file.endswith(".json"):
        target_json_file += ".json"

    training_images_array = (ctypes.c_char_p * len(training_data))()
    training_images_array[:] = [encode_str(ti) for (ti, li) in training_data]
    label_images_array = (ctypes.c_char_p * len(training_data))()
    label_images_array[:] = [encode_str(li) for (ti, li) in training_data]

    lakaseg_lib.training(
        len(training_data), training_images_array, label_images_array,
        ctypes.c_char_p(encode_str(target_json_file)), forest_size,
        max_tree_depth, testobject_tries, window_radius, number_of_threads)


def segmentieren(input_image, json_file, result_image,
                 edge_weight=5.0,
                 inference_method="maxflow",
                 gibbs_sampling_steps=2000,
                 intermediate_result_image=None,
                 ground_truth_image=None):
    """
    input_image: Pfad zum Bild, das segmentiert werden soll

    json_file: Pfad zur JSON-Datei mit dem trainierten Random Forest

    result_image: Dateiname des Ergebnisbildes (oder None, wenn es nicht
    gespeichert werden soll)

    edge_weight: je höher, desto geglätteter das Ergebnis. Es lohnt sich, an
    diesem Wert zu drehen und verschiedene Werte auszuprobieren. Übliche sind
    Werte so zwischen 3 und 15.

    inference_method: der verwendete Inferenz-Algorithmus. Entweder "maxflow"
    oder "gibbs". Ersteres ist deutlich besser.

    gibbs_sampling_steps: wenn "gibbs" als Inferenzmethode verwendet wird, gibt
    dieser Wert die Anzahl der Sampling-Durchläufe an. Je höher, desto besser
    (und glatter) i.A. das Ergebnis, aber desto länger dauert es natürlich.

    intermediate_result_image: Dateiname des Bildes mit der Ausgabe des Random
    Forests. Ein Pixel ist um so heller, je höher die Wahrscheinlichkeit, dass
    es ein Vordergrund-Pixel ist. Bei None wird das Bild nicht gespeichert.

    ground_truth_image: Dateiname des Label-Bildes mit den wahren Labels für
    das Eingabebild. Es wird mit dem Segmentierungsergebnis verglichen und das
    Ergebnis als F-Maß in eine Datei geschrieben. Eventuell nützlich um die
    Leistung zu testen. Bei None wird nichts dergleichen gemacht.
    """

    if result_image is not None and "." not in result_image:
        result_image += ".png"
    if intermediate_result_image is not None and "." not in intermediate_result_image:
        intermediate_result_image += ".png"
    if not json_file.endswith(".json"):
        json_file += ".json"

    im = 0 if inference_method == "maxflow" else 1

    iri = None if intermediate_result_image is None else ctypes.c_char_p(
        encode_str(intermediate_result_image))
    gti = None if ground_truth_image is None else ctypes.c_char_p(
        encode_str(ground_truth_image))

    lakaseg_lib.inference(ctypes.c_char_p(
        encode_str(input_image)), ctypes.c_char_p(encode_str(json_file)),
                          ctypes.c_char_p(encode_str(result_image)),
                          ctypes.c_double(edge_weight), im, iri, gti,
                          gibbs_sampling_steps)
