# Jubea_Projekt
Bauen eines Bildklassifikators für Jubaea Chilensis, eine vom aussterben bedrohte Palmenart. 
Gerade ist es nur ein Binärerklassifikator, soll in der Zukunft auf andere Arten erweitert werden.

In Dateien sind ein paar Bilder zum Trainieren/Testen. Main.ipynb war der erste Entwurf, der auf der CPU trainiert.
classificator_resnet.ipynb ist die neuere bessere Variante, die mit GPU trainiert und höhere Acc erreicht.

Jubaea_model.pth ist das beste gespeicherte Modell von Main.ipynb. ~75%

Jubaea_resnet.pth ist das beste gespeicherte Modell von classificator.ipynb. ~98%
