import sklearn
import yfinance as yf
import datetime as dt
from dateutil.relativedelta import relativedelta as rdt
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor as mlpr
from tkinter import *


def _quit():
    root.quit()  # stops mainloop
    root.destroy()


root = Tk()
root.title("AI Stock Prediction")
graphCheckbutton_variable = IntVar()

# make this object oriented. Each grid section can have its own frame.

mainFrame = Frame(root)
mainFrame.grid(row=2, padx=50, pady=30)
title = Label(mainFrame, text="Stock Ticker")
title.grid(row=0, columnspan=3)
tickerEntry = Entry(mainFrame, width=10)
tickerEntry.grid(row=1, columnspan=3)

dateLabel = Label(mainFrame, text="(Optional) Beginning Date to Analyze MM/DD/YYYY")
dateLabel.grid(row=2, columnspan=3)

dateFrame = Frame(mainFrame)

dayEntry = Entry(dateFrame, width=5)
monthEntry = Entry(dateFrame, width=5)
yearEntry = Entry(dateFrame, width=10)
dayEntry.grid(row=0, column=0)
monthEntry.grid(row=0, column=1)
yearEntry.grid(row=0, column=2)

dateFrame.grid(row=3, columnspan=3)

spaceLabel = Label(mainFrame, text="")
spaceLabel.grid(row=4, columnspan=3)

graphCheckbutton = Checkbutton(mainFrame, text="Show Graph", var=graphCheckbutton_variable)
graphCheckbutton.grid(row=5, columnspan=3)

runButton = Button(mainFrame, text="Run Prediction For Tomorrow", bg="yellow", command=_quit)
runButton.grid(row=6, columnspan=3)

graphFrame = Frame(root)
graphFrame.grid(row=1)

resultFrame = Frame(root)
resultFrame.grid(row=0)
tickerLabel = Label(resultFrame, text="Input Ticker")
tickerLabel.grid(row=0, columnspan=2)
algo_title1 = Label(resultFrame, text="Algorithm 1:")
algo_title1.grid(row=1, column=0)
result1 = Label(resultFrame, text="")
result1.grid(row=1, column=1)
algo_title2 = Label(resultFrame, text="Algorithm 2:")
algo_title2.grid(row=2, column=0)
result2 = Label(resultFrame, text="")
result2.grid(row=2, column=1)

root.mainloop()
