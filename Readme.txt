Lakaseg (kurz für "Landkarten segmentieren") ist der praktische Teil meiner
Diplomarbeit "Semantische Segmentierung historischer topographischer Karten".

Eventuelle Updates wird es unter https://github.com/EinfachToll/Lakaseg geben


Installation
============

- benötigt ImageMagick oder GraphicsMagick (unter Linux i.A. vorinstalliert,
  für Windows liegt ein Installer für GraphicsMagick bei (in 3rd_party/).)

- Für Linux oder Cygwin gibt es ein Makefile. Erstellen der Bibliothek mit
  'make'. Erstellen einer Binärdatei mit 'make bin'.

- Für Visual Studio liegt eine Solution-Datei vor (Lakaseg.sln). Die Datei ist
  ursprünglich für Visual Studio 2008. Falls die .sln-Datei nicht kompatibel
  mit der verwendeten Visual-Studio-Version ist, hier eine kleine Anleitung zum
  Anlegen des Projekts:
    für eine Exe-Datei:
    - Projekt anlegen
    - lakaseg.cpp zu dem source-Ordner hinzufügen
    - Projekteigenschaften -> C/C++ -> General -> Additional Include
      Directories -> 3rd_party/ reinschreiben
    - evtl. C/C++ -> Optimization -> Optimization -> /O2 einschalten
    - für OpenMP bei C/C++ -> Command Line -> Additional options /openmp dazu

    für die DLL:
    - Projekt anlegen, dabei bei Application Settings das Kästchen Empty
      project ankreuzen und bei Application type den Knubbel DLL
    - ansonsten wie oben


Benutzung
=========

- Zuerst muss mit Trainingsbildern ein Random Forest trainiert und gespeichert
  werden. Mit diesem kann man andere Kartenbilder segmentieren.

- Wenn man das Programm als Exe erstellt hat, kann man es auf der Kommandozeile
  benutzen. Hinweise zur Verwendung mit 'lakaseg.exe -h' (Windows) oder
  './lakaseg -h' (Linux)

- Empfohlen ist die Benutzung als Bibliothek unter Verwendung des Python-Moduls
  lakaseg.py (lauffähig mit Python 2 und Python 3).
    - die Pfade zu Ein- und Ausgabebildern und zur .json-Datei müssen unter
      Windows eventuell absolute Pfade sein statt relative


Training
--------

- Man kann den Random Forest mit mehreren Bildern trainieren. Dabei muss es zu
  jedem Kartenbild ein Bild mit Labels geben.

- die Trainingsbilder sollten Grauwertbilder sein, denn es wird nur der R-Kanal
  davon verwendet.

- Ein Labelbild muss natürlich die gleiche Größe wie das dazugehörige
  Kartenbild haben. Auch hier wird nur der R-Kanal verwendet.
  Jedes Labelbild muss genau zwei Farben enthalten, wobei die hellere für den
  (interessanten) Vorder- und die dunklere für den (uninteressanten)
  Hintergrund steht.
  Wenn man mit mehreren Bildern trainiert, müssen die Label-Farben in allen
  Bildern gleich sein.
  Die Grauwerte dieser Labels müssen > 0 sein.
  Schwarze Pixel (also Grauwert = 0) bedeuten, dass es an diesem Pixel kein
  Label gibt, solche Pixel werden also beim Training ignoriert.

Beispiel:

  ```python
  import lakaseg

  trainingsdaten = [("Testdaten/bk1.png", "Testdaten/bk1_labels.png"),
                    ("Testdaten/bk2.png", "Testdaten/bk2_labels.png")]

  lakaseg.trainieren(trainingsdaten, "forest.json", forest_size=7,
           max_tree_depth=8, testobject_tries=300, number_of_threads=4)
  ```



Segmentieren
------------

Beispiel:

  ```python
  import lakaseg

  lakaseg.segmentieren("Testdaten/bk3.png", "forest.json", "ergebnis.png")
  ```

- die Pixel am Rand des Bildes können keinem Label zugeordnet werden, weil ja
  für die Klassifikation eines Pixels die Pixel in dessen Nachbarschaft
  verwendet werden. Diese Pixel werden im Ergebnisbild schwarz.


Lizenz
======

Für die Lizenzen der benutzten Bibliotheken siehe 3rd_party/

The MIT License (MIT)

Copyright (c) 2016 Daniel Schemala

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
