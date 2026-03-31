import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

df = pd.read_csv('analysisdoc.csv', sep=',')

params = [['ts_len_avg',12], ['ts_count_total',21], ['class_count',13], ['event_count_total',25]]

for i in params:
    filterparameter = i[0]
    max_power = i[1]

    # 1. Daten filtern: Nur 'class_count'
    df_classes = df[df['parameter'] == filterparameter]
    values = df_classes['value']
    # 2. Zweierpotenzen für die Bins generieren (z. B. von 2^0 bis 2^16)
    # Das erzeugt die Liste: [1, 2, 4, 8, 16, 32, ..., 65536]

    bins = [2**i for i in range(max_power + 1)]
    # 3. Plot erstellen
    fig, ax = plt.subplots(figsize=(10, 6))
    # Histogramm mit den neuen Bins zeichnen
    ax.hist(values, bins=bins, edgecolor='black', color='lightgreen')
    # 4. X-Achse auf logarithmisch umstellen, aber dieses Mal mit Basis 2!
    ax.set_xscale('log', base=10)
    # Beschriftungen
    ax.set_title('Verteilung der '+filterparameter+' (Zweierpotenzen)')
    ax.set_xlabel(filterparameter+' (Größenordnung: 1, 2, 4, 8, 16...)')
    ax.set_ylabel('Anzahl der Datensätze')
    # 5. Formatierung der X-Achse
    # Zeigt normale Zahlen statt wissenschaftlicher Notation (z.B. 2^3)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    # Optional: Zwingt Matplotlib dazu, exakt bei unseren Bins einen Strich (Tick) zu setzen
    ax.set_xticks(bins)
    # Falls die Zahlen auf der X-Achse überlappen, rotieren wir sie leicht
    plt.xticks(rotation=45)
    plt.tight_layout() # Sorgt dafür, dass nichts abgeschnitten wird
    plt.show()