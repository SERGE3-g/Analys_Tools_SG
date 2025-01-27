import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

from Customer_Churn_Analysis_System import ChurnAnalyzer


class ChurnAnalysisTab(ttk.Frame):
    def __init__(self, notebook, current_user, user_role):
        super().__init__(notebook)
        self.current_user = current_user
        self.user_role = user_role

        self.analyzer = ChurnAnalyzer()
        self.data = None
        self.processed_data = None

        self.create_widgets()

    def create_widgets(self):
        # Menu principale
        menu_frame = ttk.Frame(self)
        menu_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(menu_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(menu_frame, text="Generate Mock Data", command=self.generate_mock_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(menu_frame, text="Train Model", command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(menu_frame, text="Analyze Segments", command=self.analyze_segments).pack(side=tk.LEFT, padx=5)
        ttk.Button(menu_frame, text="Generate Report", command=self.generate_report).pack(side=tk.LEFT, padx=5)

        # Notebook per diverse visualizzazioni
        self.sub_notebook = ttk.Notebook(self)
        self.sub_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab per i dati
        self.data_frame = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.data_frame, text="Data View")

        # Tab per le metriche
        self.metrics_frame = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.metrics_frame, text="Metrics")

        # Tab per i grafici
        self.plots_frame = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.plots_frame, text="Plots")

        # Tab per le raccomandazioni
        self.recommendations_frame = ttk.Frame(self.sub_notebook)
        self.sub_notebook.add(self.recommendations_frame, text="Recommendations")

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5)

    def load_data(self):
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.data = pd.read_csv(filename)
                self.show_data()
                self.status_var.set(f"Loaded data from {filename}")
            except Exception as e:
                logging.error(f"Error loading churn data: {str(e)}")
                messagebox.showerror("Error", f"Error loading data: {str(e)}")

    def generate_mock_data(self):
        try:
            self.data = self.analyzer.generate_mock_data(1000)
            self.show_data()
            self.status_var.set("Generated mock data")
        except Exception as e:
            logging.error(f"Error generating mock data: {str(e)}")
            messagebox.showerror("Error", f"Error generating data: {str(e)}")

    def show_data(self):
        # Clear existing widgets
        for widget in self.data_frame.winfo_children():
            widget.destroy()

        # Create treeview for data display
        tree = ttk.Treeview(self.data_frame)
        tree.pack(fill=tk.BOTH, expand=True)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.data_frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)

        # Configure columns
        columns = list(self.data.columns)
        tree["columns"] = columns
        tree["show"] = "headings"

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        # Add data
        for idx, row in self.data.head(100).iterrows():
            tree.insert("", "end", values=list(row))

    def train_model(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please load or generate data first")
            return

        try:
            self.status_var.set("Processing data...")
            self.update()

            self.processed_data = self.analyzer.preprocess_data(self.data)
            X = self.processed_data.drop('Churn', axis=1)
            y = self.processed_data['Churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.status_var.set("Training model...")
            self.update()

            self.analyzer.train_model(X_train, y_train)
            results = self.analyzer.evaluate_model(X_test, y_test)

            self.show_metrics(results)
            self.status_var.set("Model training completed")

        except Exception as e:
            logging.error(f"Error training churn model: {str(e)}")
            messagebox.showerror("Error", f"Error training model: {str(e)}")

    def show_metrics(self, results):
        # Clear existing widgets
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

        # Create text widget for metrics display
        text = tk.Text(self.metrics_frame, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True)

        # Add metrics
        text.insert(tk.END, "Model Performance Metrics\n\n")
        text.insert(tk.END, f"Accuracy: {results['accuracy']:.4f}\n")
        text.insert(tk.END, f"ROC AUC: {results['roc_auc']:.4f}\n")
        text.insert(tk.END, f"\nClassification Report:\n{results['classification_report']}\n")

        # Make text read-only
        text.configure(state='disabled')

    def analyze_segments(self):
        if self.processed_data is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        try:
            segment_analysis = self.analyzer.analyze_segment_churn(self.processed_data)
            self.show_segment_analysis(segment_analysis)
        except Exception as e:
            logging.error(f"Error analyzing segments: {str(e)}")
            messagebox.showerror("Error", f"Error analyzing segments: {str(e)}")

    def show_segment_analysis(self, segment_analysis):
        # Clear existing widgets
        for widget in self.plots_frame.winfo_children():
            widget.destroy()

        # Create figure
        fig = Figure(figsize=(12, 8))
        canvas = FigureCanvasTkAgg(fig, master=self.plots_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add plots
        self.analyzer.plot_segment_analysis(self.processed_data)

    def generate_report(self):
        if self.processed_data is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        try:
            report = self.analyzer.generate_report(self.data)

            # Save report
            filename = filedialog.asksaveasfilename(
                defaultextension=".md",
                filetypes=[("Markdown files", "*.md"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w') as f:
                    f.write(report)
                self.status_var.set(f"Report saved to {filename}")

        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            messagebox.showerror("Error", f"Error generating report: {str(e)}")

    def search(self, term):
        """Implement search functionality for the churn analysis tab"""
        results = []
        if self.data is not None:
            # Search in data columns
            for col in self.data.columns:
                if term.lower() in col.lower():
                    results.append(f"Column found: {col}")

            # Search in metrics if available
            if hasattr(self, 'latest_metrics'):
                if term.lower() in str(self.latest_metrics).lower():
                    results.append("Found in metrics analysis")

        return results