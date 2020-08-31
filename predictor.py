# Almutwakel Hassan
# Machine Learning Stock Trading Algorithm
# test key botpass123
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

# variables
matplotlib.use("TkAgg")
red = (1, 0, 0, 0.5)
green = (0, 1, 0, 0.5)
classifications = ['Up', 'Down']


class PredictionData:
    # stock data
    ticker = "###"
    data = None
    # analysis data
    difference = []  # difference between close and open
    range = []  # absolute value difference between high and low
    volume = []  # number of transactions that day
    avg_volume = 0
    trend = 0  # slope from 3 months ago to now
    # training data results
    results = []
    results_binary = []

    def __init__(self, ticker, start_date=(dt.date.today() + rdt(months=-3)).strftime('%Y-%m-%d'),
                 end_date=dt.date.today().strftime('%Y-%m-%d')):
        # get the historical prices for this ticker
        self.box_figure = None
        self.base = None
        self.data = yf.Ticker(ticker).history(period='1d', start=start_date, end=end_date)
        self.ticker = ticker
        self.difference = []
        self.range = []
        self.volume = []
        self.avg_volume = 0
        self.variables = []
        self.results = []
        for day in range(len(self.data)):
            self.difference.append(self.data["Close"][day] - self.data["Open"][day])
            self.range.append(abs(self.data["High"][day] - self.data["Low"][day]))
            vol = self.data["Volume"][day]
            self.volume.append(vol)
            self.avg_volume += vol
        self.avg_volume = int(self.avg_volume / len(self.data))
        y = (self.data["Close"][-1] + self.data["Close"][-2] + self.data["Close"][-3]) / 3
        x = (self.data["Close"][0] + self.data["Close"][1] + self.data["Close"][2]) / 3
        self.trend = y / x
        for dif in self.difference[3:]:

            # regression method
            self.results.append(dif)
            # classification method
            if dif > 0:
                self.results_binary.append(1)
            else:
                self.results_binary.append(0)
        self.variables = list(
            zip(self.difference[2:-1], self.difference[1:-2], self.difference[0:-3], self.range[2:-1], self.range[1:-2],
                self.range[0:-3], self.volume[2:-1], self.volume[1:-2], self.volume[0:-3]))
        self.tomorrow = list(
            zip(self.difference[-1:], self.difference[-1:], self.difference[-1:], self.range[-1:], self.range[-1:],
                self.range[-1:], self.volume[-1:], self.volume[-1:], self.volume[-1:]))

    def box_plot(self):
        # only open, close, high, low
        df_box = self.data[['Open', 'Close', 'High', 'Low']]
        items = len(df_box)
        self.box_figure = plt.figure()
        # self.base = plt.figure()
        self.box_figure.canvas.set_window_title('Boxplot for ' + self.ticker)
        no_median_line = dict(linestyle='-.', linewidth=0, color='black')
        boxes = plt.boxplot(df_box, widths=1, positions=range(1, 2 * items + 1)[::2], whis=1000, showcaps=False,
                            patch_artist=True, showmeans=False, meanline=False, medianprops=no_median_line)
        for i in range(items):
            if df_box['Open'][i] < df_box['Close'][i]:
                color = green
            else:
                color = red
            boxes['boxes'][i].set_facecolor(color)
        # plt.show()

    def graph(self):
        self.base.canvas.set_window_title('Price vs Volume for ' + self.ticker)
        plt.subplot(211)
        plt.plot(self.data['Open'])
        plt.plot(self.data['Close'])
        plt.ylabel('Price')
        plt.subplot(212)
        plt.plot(self.data['Volume'])
        plt.ylabel('Volume')
        plt.show()

    def get_all(self):
        # print(self.data[['Open', 'Close', 'Low', 'High', 'Volume']])
        # print(self.difference)
        # print(self.range)
        # print(self.volume)
        # print(self.avg_volume)
        # print(self.trend)
        return self.data[['Open', 'Close', 'Low', 'High',
                          'Volume']], self.difference, self.range, self.volume, self.avg_volume, self.trend

    # def train_SVM(self):
        # print('start')
        # Support Vector Machine model training
        # , [self.avg_volume for _ in range(len(self.data))], [self.trend for _ in range(len(self.data))]
        # x = self.variables
        # y = self.results
        # y_binary = self.results_binary
        # print('checkpoint 1')

        # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y_binary, test_size=0.2)
        # x_train2, x_test2, y_train2, y_test2 = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
        # print('checkpoint 2')

        # print(x_train, y_train)
        # reg = svm.SVR(kernel="poly", degree=2, C=2)
        #        clf = svm.SVC(kernel="linear", C=2, verbose=True, probability=True)
        # print('checkpoint 3')
        # print(x_train2, y_train2)
        # reg.fit(x_train2, y_train2)
        #        clf.fit(x_train, y_train)
        # print('checkpoint 4')
        #        with open("predictor_SVC.models", "wb") as f:
        #            models.dump(clf, f)

        # print('checkpoint 5')
        #        with open("predictor_SVM.models", "wb") as f:
        #            pickle_in = open("predictor_SVM.models", "rb")
        #            clf_loaded = models.load(pickle_in)
        # y_prediction = reg.predict(x_test2)
        #        acc = metrics.accuracy_score(y_prediction, y_test)
        # classifications_predictions = []
        # print(y_prediction, y_test2)
        # print('end')

    def train_linear_regression(self, retrain=False):
        saved = None
        # Linear regression model training
        best = 0
        x = self.variables
        y = self.results
        # if retrain:
        linear = linear_model.LinearRegression()
        for _ in range(100):
            # create best fit line based on data
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
            linear = linear_model.LinearRegression()
            linear.fit(x_train, y_train)

            # accuracy score
            acc = linear.score(x_test, y_test)
            # print(acc)

            # save it
            if acc > best:
                best = acc
                saved = linear
                # with open("models/predictor_LIN.pickle", "wb") as f:
                #    pickle.dump(linear, f)

        x_test, y_test = x, y
        # pickle_in = open("models/predictor_LIN.pickle", "rb")
        # linear = pickle.load(pickle_in)
        linear = saved
        predictions = linear.predict(x_test)
        correct = 0
        wrong = 0
        for x in range(len(predictions)):
            if predictions[x] > 0 and y_test[x] > 0:
                correct += 1
            elif predictions[x] < 0 and y_test[x] < 0:
                correct += 1
            else:
                wrong += 1
        #    print(predictions[x], x_test[x], y_test[x])
        # print(str(correct) + " correct " + str(wrong) + " wrong")
        percent = correct / (correct + wrong)
        # print(percent, "%")
        # print(linear.score(x_test, y_test))

        prediction = linear.predict(self.tomorrow)
        return prediction, percent

    def train_sklearn_neuralnet(self, retrain=False):
        saved = None
        x = self.variables
        y = self.results

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

        best = -100000000000000
        neuralnet = mlpr(solver='adam', hidden_layer_sizes=(4, 4, 4))

        if retrain:
            for _ in range(100):
                # create best fit line based on data

                neuralnet = mlpr(solver='adam', hidden_layer_sizes=(4, 4, 4))
                x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
                neuralnet.fit(x_train, y_train)

                # accuracy score
                acc = neuralnet.score(x_test, y_test)
                # print(acc)

                # save it
                if acc > best:
                    best = acc
                    saved = neuralnet
                    # with open("models/predictor_NRL.pickle", "wb") as f:
                    #    pickle.dump(neuralnet, f)
        else:
            x_test, y_test = x, y
        # pickle_in = open("models/predictor_NRL.pickle", "rb")
        # neuralnet = pickle.load(pickle_in)
        neuralnet = saved
        predictions = neuralnet.predict(x_test)
        correct = 0
        wrong = 0
        x_test, y_test = x, y
        for x in range(len(predictions)):
            if predictions[x] > 0 and y_test[x] > 0:
                correct += 1
            elif predictions[x] < 0 and y_test[x] < 0:
                correct += 1
            else:
                wrong += 1
            # print(predictions[x], x_test[x], y_test[x])
        print(str(correct) + " correct " + str(wrong) + " wrong")
        percent = correct / (correct + wrong)
        # print(percent, "%")
        # print(neuralnet.score(x_test, y_test))

        prediction = neuralnet.predict(self.tomorrow)

        return prediction, percent

    # def train_tf_neuralnet(self, retrain=False):
    #    # x = list(zip(self.difference, self.range, self.volume, [self.trend for _ in range(len(self.difference) + 1)], [self.avg_volume for _ in range(len(self.difference) + 1)]))[:-1]
    #    x = self.variables
    #    y = self.results
    #
    #    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    #
    #    if retrain:
    #        model = keras.Sequential([
    #            keras.layers.Dense(9, input_dim=9, kernel_initializer='normal'),
    #            keras.layers.Dense(6, activation="relu"),
    #            keras.layers.Dense(3, activation="relu"),
    #            # create the output layer with softmax activation function, which shows probability
    #            keras.layers.Dense(1)
    #        ])

            # create loss function to optimize data
    #        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

            # train the model
    #        model.fit(x_train, y_train, epochs=80)
            # save the model
    #        model.save('models/predictor_KRS')
    #    else:
    #        x_test, y_test = x, y

        # load model
    #    model = keras.models.load_model("models/predictor_KRS")
        # test the model
    #    predictions = model.predict(x)
    #    correct = 0
    #    wrong = 0
    #    for p in range(len(predictions)):
    #        if predictions[p] > 0 and y[p] > 0:
    #            correct += 1
    #        elif predictions[p] < 0 and y[p] < 0:
    #            correct += 1
    #        else:
    #            wrong += 1
            # print(predictions[p], x[p], y[p])
        # print(str(correct) + " correct " + str(wrong) + " wrong")
    #    percent = correct / (correct + wrong)
        # print(percent, "%")

        # use it to predict tomorrow
    #    prediction = model.predict(self.tomorrow)

    #    return prediction, percent

    def predict_tomorrow(self, retrain=True):
        pred1, confidence1 = self.train_linear_regression(retrain)
        # (SVM algorithm non-functional)
        # pred2, confidence2 = self.train_SVM(retrain)
        pred3, confidence3 = self.train_sklearn_neuralnet(retrain)
        # (TF algorithm low accuracy)
        # pred4, confidence4 = self.train_tf_neuralnet(retrain)
        if pred1[0] >= 0:
            pred1 = "+" + str(round(pred1[0], 2))
        else:
            pred1 = str(round(pred1[0] * 100, 2))
        if pred3[0] >= 0:
            pred3 = "+" + str(round(pred3[0], 2))
        else:
            pred3 = str(round(pred3[0] * 100, 2))
        str(round(confidence3 * 100, 2))
        print("Prediction for", self.ticker, "for", (dt.date.today() + rdt(days=+1)).strftime('%Y-%m-%d'))
        algo_title1['text'] = "Linear Regression yields prediction: " + pred1
        result1['text'] = "Confidence: " + str(round(confidence1 * 100, 2)) + "%"
        algo_title2['text'] = "Neural Network yields prediction: " + pred3
        result2['text'] = "Confidence: " + str(round(confidence3 * 100, 2)) + "%"
        # print("Tensorflow Keras Neural Network yields prediction:", pred4, confidence4)


class FramedLabel(object):
    def __init__(self, text, center_x, center_y, height, width):
        self.frame = Frame(mainFrame, width=width, height=height, pady=height / 2, padx=width / 2, expand=True)
        self.label = Label(self.frame, text=text)
        self.label.place(x=center_x - width / 2, y=center_y - height / 2, width=width, height=height)


class FramedEntry(object):
    def __init__(self, height, width, center_x=0, center_y=0):
        self.frame = Frame(mainFrame, width=width, height=height, pady=0, padx=width / 2, expand=True)
        self.label = Entry(self.frame)
        if center_x == 0 and center_y == 0:
            self.label.place(width=width, height=height)
        else:
            self.label.place(x=center_x - width / 2, y=center_y - height / 2, width=width, height=height)

    def add_entry(self, height, width, x_offset, y_offset):
        self.label = Entry(self.frame)
        self.label.place(x=x_offset, y=y_offset, height=height, width=width)


def run_button(retrain=True):
    ticker = tickerEntry.get()
    try:
        if dayEntry.get() == "" or monthEntry.get() == "" or yearEntry.get() == "":
            pred_obj = PredictionData(ticker)
        else:
            date = yearEntry.get() + "-" + dayEntry.get() + "-" + monthEntry.get()
            try:
                pred_obj = PredictionData(ticker, date)
            except:
                tickerLabel['text'] = "Invalid Date: " + date
                return None

        tickerLabel['text'] = "Results for Ticker: " + ticker
    except ZeroDivisionError:
        tickerLabel['text'] = "Invalid Ticker: " + ticker
        return None
    # x = pred_obj.variables
    # y = pred_obj.results
    # print("X:", len(x), x)
    # print("Y:", len(y), y)
    pred_obj.predict_tomorrow(retrain)
    pred_obj.box_plot()
    if graphCheckbutton_variable.get():
        canvas = FigureCanvasTkAgg(pred_obj.box_figure, master=graphFrame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0)
    del pred_obj


def _quit():
    root.quit()  # stops mainloop
    root.destroy()


# my_test.print_all()
# my_test.box_plot()
# my_test.train_linear_regression(True)
# print(my_test.train_tf_neuralnet(True))
# my_test.train_sklearn_neuralnet(True)

root = Tk()
root.title("AI Swing Trader Stock Prediction")
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

runButton = Button(mainFrame, text="Run Prediction For Tomorrow", bg="yellow", command=run_button)
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

root.protocol('WM_DELETE_WINDOW', _quit)
root.mainloop()
