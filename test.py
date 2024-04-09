# from matplotlib import pyplot as plt
# from sklearn.metrics import confusion_matrix , classification_report
# import pandas as pd
# import seaborn as sns




# def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
#     """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
#     Arguments
#     ---------
#     confusion_matrix: numpy.ndarray
#         The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
#         Similarly constructed ndarrays can also be used.
#     class_names: list
#         An ordered list of class names, in the order they index the given confusion matrix.
#     figsize: tuple
#         A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
#         the second determining the vertical size. Defaults to (10,7).
#     fontsize: int
#         Font size for axes labels. Defaults to 14.
        
#     Returns
#     -------
#     matplotlib.figure.Figure
#         The resulting confusion matrix figure
#     """
#     df_cm = pd.DataFrame(
#         confusion_matrix, index=class_names, columns=class_names, 
#     )
#     fig = plt.figure(figsize=figsize)
#     try:
#         heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
#     except ValueError:
#         raise ValueError("Confusion matrix values must be integers.")
#     heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
#     heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
#     plt.figure()
#     plt.title("Confusion Matrix")
#     plt.ylabel('Truth')
#     plt.xlabel('Prediction')
#     plt.show()
# truth =      ["Dog","Not a dog","Dog","Dog",      "Dog", "Not a dog", "Not a dog", "Dog",       "Dog", "Not a dog"]
# prediction = ["Dog","Dog",      "Dog","Not a dog","Dog", "Not a dog", "Dog",       "Not a dog", "Dog", "Dog"]
# cm = confusion_matrix(truth,prediction)
# print_confusion_matrix(cm,["Dog","Not a dog"])


import threading
import time
from tqdm import tqdm

def func(t):
    print("Start")
    time.sleep(t)
    print("End")


def main():
    symbols = ['⣾', '⣷', '⣯', '⣟', '⡿', '⢿', '⣻', '⣽']
    i = 0
    d=0
    while d<1000:
        i = (i + 1) % len(symbols)
        print('\r\033[K%s loading...' % symbols[i], flush=True, end='')
        time.sleep(0.05)
        d+=1
    print("Done")


c=threading.Thread(target=func, args=(5,))
c.start()
c2=threading.Thread(target=main)
c2.start()
    
# main()