#!/usr/bin/env python
# coding: utf-8

import lakaseg


testdaten = ("../Testdaten/bk1.png", "../Testdaten/labels_bk1_ausschnitt.png")

lakaseg.trainieren(testdaten, "forest.json")

lakaseg.segmentieren("../Testdaten/bk1_sw.png", "forest.json", "ziel.png", intermediate_result_image="pot.png")
