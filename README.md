Projekt przedstawia kompletny proces budowy, trenowania i analizowania modelu konwolucyjnej sieci neuronowej przeznaczonej do klasyfikacji obrazów ze zbioru CIFAR-10. 
Całość została zrealizowana przy użyciu bibliotek Keras oraz TensorFlow i opiera się na dobrych praktykach stosowanych w deep learningu, takich jak:
batch normalization, regularizacja wag L2, warstwy dropout, a także mechanizmy stabilizujące uczenie, czyli EarlyStopping oraz ReduceLROnPlateau.

W projekcie wykorzystano zbiór danych CIFAR-10, który zawiera 60 tysięcy kolorowych obrazów o wymiarach 32×32 piksele, podzielonych na dziesięć kategorii obejmujących obiekty naturalne i pojazdy. 
Zbiór został rozdzielony na część treningową oraz testową, odpowiednio 50 tysięcy i 10 tysięcy elementów. Dane są wstępnie normalizowane, a etykiety przekształcane do formatu one-hot encoding, co przygotowuje je do pracy z klasyfikatorem softmax.

Zaprojektowana sieć składa się z trzech bloków konwolucyjnych. Każdy blok obejmuje warstwy Conv2D, batch normalization, aktywację ReLU, dropout oraz pooling typu MaxPooling2D. 
Po przejściu przez ekstrakcję cech obraz trafia do warstwy spłaszczającej, a następnie do gęstej warstwy Dense z 256 neuronami i aktywacją ReLU. 
Na końcu model wykorzystuje warstwę softmax, która dokonuje klasyfikacji do jednej z dziesięciu kategorii. 
W trakcie trenowania zastosowano inicjalizację wag He Normal, optymalizator Adam z learning rate równym 0.0007 oraz regularizację L2 o wartości 5e-4, która pozwala ograniczyć przeuczenie modelu. 
Proces uczenia przewidziano maksymalnie na sto epok, przy czym EarlyStopping zatrzymuje go wcześniej, jeśli model przestaje poprawiać wyniki walidacyjne. 
W trakcie treningu automatycznie redukowany jest również learning rate w momentach stagnacji.

Najlepszy osiągnięty w trakcie walidacji wynik funkcji straty to 0.5222. Po zakończonym uczeniu model poddawany jest ocenie na zbiorze testowym, a wszystkie kluczowe statystyki są wypisywane w konsoli. 
Projekt generuje również wizualizacje, które znacznie ułatwiają analizę działania modelu: 
wykresy przebiegu strat i dokładności, przegląd przykładowych błędnie sklasyfikowanych obrazów wraz z przewidywaniami oraz macierz pomyłek pozwalająca sprawdzić, które klasy sprawiają modelowi największe trudności.

