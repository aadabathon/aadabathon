import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
 
# Tooltip Class Definition
class Tooltip:
    """
    It creates a tooltip for a given widget as the mouse goes on it.
 
    Usage:
        tooltip = Tooltip(widget, "This is the tooltip text")
    """
 
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)
 
    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        # Position the tooltip
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # Create a toplevel window
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)  # Removes all window decorations
        tw.wm_geometry("+%d+%d" % (x, y))
        # Add a label to the toplevel window
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)
 
    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()
 
def simulate_trades(num_trades, win_rate, avg_rr, percent_risked, starting_balance, compounded):
    """
    Simulates a series of trades and returns the cumulative equity curve and trade details.
 
    Returns:
    - equity (numpy.ndarray): Cumulative equity after each trade.
    - outcomes (list): List of trade outcomes (1 for win, 0 for loss).
    - trade_profits (list): List of profit/loss amounts per trade.
    """
    equity = np.zeros(num_trades + 1)
    equity[0] = starting_balance
    outcomes = []
    trade_profits = []
 
    for i in range(1, num_trades + 1):
        current_equity = equity[i - 1]
        if compounded:
            risk_amount = (percent_risked / 100) * current_equity
        else:
            risk_amount = (percent_risked / 100) * starting_balance
 
        avg_win = avg_rr * risk_amount
        avg_loss = risk_amount
 
        # Determine win or loss
        outcome = np.random.rand()
        if outcome < win_rate:
            profit = avg_win
            outcomes.append(1)  # Win
        else:
            profit = -avg_loss
            outcomes.append(0)  # Loss
 
        trade_profits.append(profit)
        equity[i] = equity[i - 1] + profit
 
    return equity, outcomes, trade_profits
 
def calculate_streaks(outcomes):
    """
    Calculates the longest winning and losing streaks from trade outcomes.
 
    Parameters:
    - outcomes (list): List of trade outcomes (1 for win, 0 for loss).
 
    Returns:
    - max_win_streak (int): Longest consecutive wins.
    - max_loss_streak (int): Longest consecutive losses.
    """
    max_win_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
 
    for outcome in outcomes:
        if outcome == 1:
            current_win_streak += 1
            if current_win_streak > max_win_streak:
                max_win_streak = current_win_streak
            current_loss_streak = 0
        else:
            current_loss_streak += 1
            if current_loss_streak > max_loss_streak:
                max_loss_streak = current_loss_streak
            current_win_streak = 0
 
    return max_win_streak, max_loss_streak
 
class TradingSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Day Trading Risk Management Simulator")
        self.create_widgets()
 
    def create_widgets(self):
        # Create a frame for input fields
        input_frame = ttk.Frame(self.root, padding="10")
        input_frame.grid(row=0, column=0, sticky=tk.W)
 
        # Number of Trades
        ttk.Label(input_frame, text="Number of Trades:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.num_trades_var = tk.StringVar(value="100")
        self.num_trades_entry = ttk.Entry(input_frame, width=20, textvariable=self.num_trades_var)
        self.num_trades_entry.grid(row=0, column=1, pady=5)
 
        # Win Rate
        ttk.Label(input_frame, text="Win Rate (0-1):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.win_rate_var = tk.StringVar(value="0.55")
        self.win_rate_entry = ttk.Entry(input_frame, width=20, textvariable=self.win_rate_var)
        self.win_rate_entry.grid(row=1, column=1, pady=5)
 
        # Average RR
        ttk.Label(input_frame, text="Average RR:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.avg_rr_var = tk.StringVar(value="2")
        self.avg_rr_entry = ttk.Entry(input_frame, width=20, textvariable=self.avg_rr_var)
        self.avg_rr_entry.grid(row=2, column=1, pady=5)
 
        # % Risked
        ttk.Label(input_frame, text="% Risked per Trade:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.percent_risked_var = tk.StringVar(value="1")
        self.percent_risked_entry = ttk.Entry(input_frame, width=20, textvariable=self.percent_risked_var)
        self.percent_risked_entry.grid(row=3, column=1, pady=5)
 
        # Starting Balance
        ttk.Label(input_frame, text="Starting Balance:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.starting_balance_var = tk.StringVar(value="1000")
        self.starting_balance_entry = ttk.Entry(input_frame, width=20, textvariable=self.starting_balance_var)
        self.starting_balance_entry.grid(row=4, column=1, pady=5)
 
        # Number of Simulations
        ttk.Label(input_frame, text="Number of Simulations:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.num_simulations_var = tk.StringVar(value="10")
        self.num_simulations_entry = ttk.Entry(input_frame, width=20, textvariable=self.num_simulations_var)
        self.num_simulations_entry.grid(row=5, column=1, pady=5)
 
        # Compounded Toggle
        self.compounded_var = tk.BooleanVar()
        self.compounded_check = ttk.Checkbutton(
            input_frame,
            text="Compounded Equity",
            variable=self.compounded_var
        )
        self.compounded_check.grid(row=6, column=0, columnspan=2, pady=5)
 
        # Simulate Button
        self.simulate_button = ttk.Button(input_frame, text="Simulate", command=self.run_simulation)
        self.simulate_button.grid(row=7, column=0, columnspan=2, pady=10)
 
        # Create a frame for the plot
        self.plot_frame = ttk.Frame(self.root, padding="10")
        self.plot_frame.grid(row=1, column=0)
 
        # Create a frame for statistics
        self.stats_frame = ttk.Frame(self.root, padding="10")
        self.stats_frame.grid(row=2, column=0, sticky=tk.W)
 
        # Existing Labels
        self.best_result_label = ttk.Label(self.stats_frame, text="Best Result: N/A", font=('Arial', 12))
        self.best_result_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        Tooltip(self.best_result_label, "The highest final equity achieved across all simulations.")
 
        self.average_result_label = ttk.Label(self.stats_frame, text="Average Result: N/A", font=('Arial', 12))
        self.average_result_label.grid(row=0, column=1, sticky=tk.W, pady=2)
        Tooltip(self.average_result_label, "The mean final equity across all simulations.")
 
        self.worst_result_label = ttk.Label(self.stats_frame, text="Worst Result: N/A", font=('Arial', 12))
        self.worst_result_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        Tooltip(self.worst_result_label, "The lowest final equity achieved across all simulations.")
 
        # Existing Streak Labels
        self.longest_winning_streak_label = ttk.Label(self.stats_frame, text="Longest Winning Streak: N/A", font=('Arial', 12))
        self.longest_winning_streak_label.grid(row=2, column=0, sticky=tk.W, pady=2)
        Tooltip(self.longest_winning_streak_label, "The highest number of consecutive winning trades in any simulation.")
 
        self.longest_losing_streak_label = ttk.Label(self.stats_frame, text="Longest Losing Streak: N/A", font=('Arial', 12))
        self.longest_losing_streak_label.grid(row=2, column=1, sticky=tk.W, pady=2)
        Tooltip(self.longest_losing_streak_label, "The highest number of consecutive losing trades in any simulation.")
 
        self.average_winning_streak_label = ttk.Label(self.stats_frame, text="Average Winning Streak: N/A", font=('Arial', 12))
        self.average_winning_streak_label.grid(row=3, column=0, sticky=tk.W, pady=2)
        Tooltip(self.average_winning_streak_label, "The mean of the longest winning streaks from each simulation.")
 
        self.average_losing_streak_label = ttk.Label(self.stats_frame, text="Average Losing Streak: N/A", font=('Arial', 12))
        self.average_losing_streak_label.grid(row=3, column=1, sticky=tk.W, pady=2)
        Tooltip(self.average_losing_streak_label, "The mean of the longest losing streaks from each simulation.")
 
        # New Labels for Additional Stats
        self.profit_factor_label = ttk.Label(self.stats_frame, text="Profit Factor: N/A", font=('Arial', 12))
        self.profit_factor_label.grid(row=4, column=0, sticky=tk.W, pady=2)
        Tooltip(self.profit_factor_label, "The ratio of total gross profits to total gross losses.")
 
        self.expectancy_label = ttk.Label(self.stats_frame, text="Expectancy: N/A", font=('Arial', 12))
        self.expectancy_label.grid(row=4, column=1, sticky=tk.W, pady=2)
        Tooltip(self.expectancy_label, "The average expected return per trade.")
 
        self.max_drawdown_label = ttk.Label(self.stats_frame, text="Maximum Drawdown: N/A", font=('Arial', 12))
        self.max_drawdown_label.grid(row=5, column=0, sticky=tk.W, pady=2)
        Tooltip(self.max_drawdown_label, "The largest peak-to-trough decline in the equity curve.")
 
        self.largest_gain_label = ttk.Label(self.stats_frame, text="Largest Gain: N/A", font=('Arial', 12))
        self.largest_gain_label.grid(row=5, column=1, sticky=tk.W, pady=2)
        Tooltip(self.largest_gain_label, "The maximum profit from a single trade.")
 
        self.largest_loss_label = ttk.Label(self.stats_frame, text="Largest Loss: N/A", font=('Arial', 12))
        self.largest_loss_label.grid(row=6, column=0, sticky=tk.W, pady=2)
        Tooltip(self.largest_loss_label, "The maximum loss from a single trade.")
 
        self.average_profit_label = ttk.Label(self.stats_frame, text="Average Profit per Trade: N/A", font=('Arial', 12))
        self.average_profit_label.grid(row=6, column=1, sticky=tk.W, pady=2)
        Tooltip(self.average_profit_label, "The mean profit or loss generated per trade.")
 
        self.recovery_factor_label = ttk.Label(self.stats_frame, text="Recovery Factor: N/A", font=('Arial', 12))
        self.recovery_factor_label.grid(row=7, column=0, sticky=tk.W, pady=2)
        Tooltip(self.recovery_factor_label, "The ratio of net profit to maximum drawdown.")
 
    def run_simulation(self):
        # Retrieve and validate inputs
        try:
            num_trades = int(self.num_trades_var.get())
            if num_trades <= 0:
                raise ValueError("Number of trades must be a positive integer.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid number of trades: {e}")
            return
 
        try:
            win_rate = float(self.win_rate_var.get())
            if not 0 <= win_rate <= 1:
                raise ValueError("Win rate must be between 0 and 1.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid win rate: {e}")
            return
 
        try:
            avg_rr = float(self.avg_rr_var.get())
            if avg_rr <= 0:
                raise ValueError("Average RR must be a positive number.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid average RR: {e}")
            return
 
        try:
            percent_risked = float(self.percent_risked_var.get())
            if not 0 < percent_risked <= 100:
                raise ValueError("% Risked must be between 0 and 100.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid % Risked: {e}")
            return
 
        try:
            starting_balance = float(self.starting_balance_var.get())
            if starting_balance < 0:
                raise ValueError("Starting balance cannot be negative.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid starting balance: {e}")
            return
 
        try:
            num_simulations = int(self.num_simulations_var.get())
            if num_simulations <= 0:
                raise ValueError("Number of simulations must be a positive integer.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid number of simulations: {e}")
            return
 
        compounded = self.compounded_var.get()
 
        # Perform simulations
        equities = []
        max_win_streaks = []
        max_loss_streaks = []
        profit_factors = []
        expectancies = []
        max_drawdowns = []
        largest_gains = []
        largest_losses = []
        average_profits = []
        recovery_factors = []
 
        for _ in range(num_simulations):
            equity, outcomes, trade_profits = simulate_trades(
                num_trades,
                win_rate,
                avg_rr,
                percent_risked,
                starting_balance,
                compounded
            )
            equities.append(equity)
 
            # Calculate streaks
            max_win_streak, max_loss_streak = calculate_streaks(outcomes)
            max_win_streaks.append(max_win_streak)
            max_loss_streaks.append(max_loss_streak)
 
            # Calculate Profit Factor
            total_profit = sum([p for p in trade_profits if p > 0])
            total_loss = abs(sum([p for p in trade_profits if p < 0]))
            profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
            profit_factors.append(profit_factor)
 
            # Calculate Expectancy
            loss_rate = 1 - win_rate
            # Adjust expectancy calculation based on compounded or not
            if compounded:
                # Since risk_amount varies, use average risk_amount for expectancy
                average_risk_amount = percent_risked / 100 * starting_balance * (1 + avg_rr) / 2  # Approximation
            else:
                average_risk_amount = (percent_risked / 100) * starting_balance
            expectancy = (win_rate * (avg_rr * average_risk_amount)) - (loss_rate * average_risk_amount)
            expectancies.append(expectancy)
 
            # Calculate Maximum Drawdown
            running_max = np.maximum.accumulate(equity)
            drawdowns = running_max - equity
            max_drawdown = np.max(drawdowns)
            max_drawdowns.append(max_drawdown)
 
            # Calculate Largest Gain and Loss
            largest_gain = max(trade_profits) if trade_profits else 0
            largest_loss = min(trade_profits) if trade_profits else 0
            largest_gains.append(largest_gain)
            largest_losses.append(largest_loss)
 
            # Calculate Average Profit per Trade
            average_profit = np.mean(trade_profits) if trade_profits else 0
            average_profits.append(average_profit)
 
            # Calculate Recovery Factor
            net_profit = equity[-1] - starting_balance
            recovery_factor = net_profit / max_drawdown if max_drawdown != 0 else float('inf')
            recovery_factors.append(recovery_factor)
 
        # Convert to numpy array for easier manipulation
        equities = np.array(equities)
 
        # Plot the equity curves and display statistics
        self.plot_equity_curves(equities, starting_balance, compounded)
        self.display_statistics(
            equities,
            max_win_streaks,
            max_loss_streaks,
            profit_factors,
            expectancies,
            max_drawdowns,
            largest_gains,
            largest_losses,
            average_profits,
            recovery_factors
        )
 
    def plot_equity_curves(self, equities, starting_balance, compounded):
        # Clear previous plot if it exists
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
 
        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
 
        # Plot each simulation in blue with transparency
        for equity in equities:
            ax.plot(equity, color='blue', alpha=0.3)
 
        # Identify the best and worst simulations based on final equity
        final_equities = equities[:, -1]
        best_index = np.argmax(final_equities)
        worst_index = np.argmin(final_equities)
        best_equity = equities[best_index]
        worst_equity = equities[worst_index]
 
        # Plot the best simulation in green
        ax.plot(best_equity, color='green', linewidth=2, label='Best Simulation')
 
        # Plot the worst simulation in red
        ax.plot(worst_equity, color='red', linewidth=2, label='Worst Simulation')
 
        # Calculate and plot the average equity curve in orange
        average_equity = equities.mean(axis=0)
        ax.plot(average_equity, color='orange', linewidth=2, label='Average Equity')
 
        # Enhance the plot
        ax.set_title('Trading Equity Curve', fontsize=16)
        ax.set_xlabel('Number of Trades', fontsize=14)
        ax.set_ylabel('Cumulative Profit/Loss', fontsize=14)
        ax.grid(True)
        ax.axhline(starting_balance, color='black', linewidth=1, linestyle='--', label='Starting Balance')  # Horizontal line at starting balance
 
        # Create a custom legend
        ax.legend()
 
        # Add a text box indicating whether it's compounded or not
        compounding_text = "Compounded Equity" if compounded else "Fixed Risk"
        ax.text(0.95, 0.95, compounding_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
 
        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
 
    def display_statistics(self, equities, max_win_streaks, max_loss_streaks, profit_factors, expectancies, max_drawdowns, largest_gains, largest_losses, average_profits, recovery_factors):
        """
        Calculates and displays various statistics from the simulations.
 
        Parameters:
        - equities (numpy.ndarray): Array of equity curves.
        - max_win_streaks (list): List of maximum winning streaks from each simulation.
        - max_loss_streaks (list): List of maximum losing streaks from each simulation.
        - profit_factors (list): List of profit factors from each simulation.
        - expectancies (list): List of expectancies from each simulation.
        - max_drawdowns (list): List of maximum drawdowns from each simulation.
        - largest_gains (list): List of largest gains from each simulation.
        - largest_losses (list): List of largest losses from each simulation.
        - average_profits (list): List of average profits per trade from each simulation.
        - recovery_factors (list): List of recovery factors from each simulation.
        """
        final_equities = equities[:, -1]
 
        best_result = np.max(final_equities)
        average_result = np.mean(final_equities)
        worst_result = np.min(final_equities)
 
        longest_winning_streak = np.max(max_win_streaks)
        longest_losing_streak = np.max(max_loss_streaks)
        average_winning_streak = np.mean(max_win_streaks)
        average_losing_streak = np.mean(max_loss_streaks)
 
        profit_factor = np.mean(profit_factors) if profit_factors else 0
        expectancy = np.mean(expectancies) if expectancies else 0
        max_drawdown = np.max(max_drawdowns) if max_drawdowns else 0
        largest_gain = np.max(largest_gains) if largest_gains else 0
        largest_loss = np.min(largest_losses) if largest_losses else 0
        average_profit = np.mean(average_profits) if average_profits else 0
        recovery_factor = np.mean(recovery_factors) if recovery_factors else 0
 
        # Format the results to two decimal places
        self.best_result_label.config(text=f"Best Result: ${best_result:,.2f}")
        self.average_result_label.config(text=f"Average Result: ${average_result:,.2f}")
        self.worst_result_label.config(text=f"Worst Result: ${worst_result:,.2f}")
 
        self.longest_winning_streak_label.config(text=f"Longest Winning Streak: {longest_winning_streak}")
        self.longest_losing_streak_label.config(text=f"Longest Losing Streak: {longest_losing_streak}")
        self.average_winning_streak_label.config(text=f"Average Winning Streak: {average_winning_streak:.2f}")
        self.average_losing_streak_label.config(text=f"Average Losing Streak: {average_losing_streak:.2f}")
 
        self.profit_factor_label.config(text=f"Profit Factor: {profit_factor:.2f}")
        self.expectancy_label.config(text=f"Expectancy: ${expectancy:.2f}")
        self.max_drawdown_label.config(text=f"Maximum Drawdown: ${max_drawdown:,.2f}")
        self.largest_gain_label.config(text=f"Largest Gain: ${largest_gain:,.2f}")
        self.largest_loss_label.config(text=f"Largest Loss: ${largest_loss:,.2f}")
        self.average_profit_label.config(text=f"Average Profit per Trade: ${average_profit:,.2f}")
        self.recovery_factor_label.config(text=f"Recovery Factor: {recovery_factor:.2f}")
 
def main():
    root = tk.Tk()
    app = TradingSimulatorApp(root)
    root.mainloop()
 
if __name__ == "__main__":
    main()