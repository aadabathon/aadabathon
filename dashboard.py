import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

class IBApp(EWrapper, EClient):

    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.historical_data= {}

    def error(self, reqID, errorCode, errorString):
        print(f"Error {reqID} {errorCode} {errorString}")

    def historicalData(self, reqID, bar):
        if reqID not in self.historicalData:
            self.historical_data[reqID] = []
        self.historical_data[reqID].append({
            'date' : bar.date,
            'open' : bar.open,
            'close' : bar.close,
            'high' : bar.low,
            'volume' : bar.volume
        })

        def histroicalData(self, reqID, start, end):
            if not reqID:
                print("error no reqID")
            print(f"Historical data has been received for reqid {reqID}")
            
        
