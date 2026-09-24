# Jubea_Projekt
Bauen eines Bildklassifikators für Jubaea Chilensis, eine vom aussterben bedrohte Palmenart. 
Gerade ist es nur ein Binärerklassifikator, soll in der Zukunft auf andere Arten erweitert werden.

In Dateien sind ein paar Bilder zum Trainieren/Testen. Main.ipynb war der erste Entwurf, der auf der CPU trainiert.
classificator_resnet.ipynb ist die neuere bessere Variante, die mit GPU trainiert und höhere Acc erreicht.

Jubaea_model.pth ist das beste gespeicherte Modell von Main.ipynb. ~75%

Jubaea_resnet.pth ist das beste gespeicherte Modell von classificator.ipynb. ~98%

Es wurde Frozen Backbone mit Resnet getestet. Hat zu einer verschlechterung der Acc und des Losses geführt.
Idee wurde dementsprechend verworfen.

Die gespeicherten vorhin geneannten Modelle waren Fehlerhaft. Deswegen wurde das trainieren und testen neu gestaltet,
indem man nur am Ende einmal testet und nicht bei jeder Epoch. Stattdessen nach jedem Training kommt eine Validation.

