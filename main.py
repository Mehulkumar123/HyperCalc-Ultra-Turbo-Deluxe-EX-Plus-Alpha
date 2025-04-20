import tkinter as tk
from tkinter import ttk, messagebox, colorchooser, filedialog, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sympy as sp
import json
import csv
import random
import os
from datetime import datetime
from copy import deepcopy
import logging
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

# Setup logging
logging.basicConfig(filename='graph_tool.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Expression:
    def __init__(self, expr, plot_type='function', color='blue', style='solid', enabled=True, note='', group=''):
        self.expr = expr
        self.plot_type = plot_type
        self.color = color
        self.style = style
        self.enabled = enabled
        self.note = note
        self.opacity = 1.0
        self.group = group
        self.cached_func = None
        self.error_data = None  # For error bars in point plots

    def validate(self, robust_mode=False):
        x, y, t, theta, z, I = sp.symbols('x y t theta z I')
        try:
            if self.plot_type == 'point':
                eval(self.expr, {'np': np})
            elif self.plot_type in ['implicit', 'slope_field', 'conic']:
                sp.sympify(self.expr.replace('=', '-'))
            elif self.plot_type == 'piecewise':
                parts = self.expr.split(':')
                for part in parts:
                    sp.sympify(part.strip('{}'))
            elif self.plot_type in ['derivative', 'integral']:
                sp.sympify(self.expr.split('=')[1]).diff(x) if self.plot_type == 'derivative' else sp.sympify(self.expr.split('=')[1]).integrate(x)
            elif self.plot_type == 'complex':
                expr = sp.sympify(self.expr.split('=')[1] if '=' in self.expr else self.expr)
                if not expr.has(z):
                    return False, "Complex plot requires 'z' (e.g., z = x + I*y)."
                # Test evaluation
                sp.lambdify(z, expr, modules=['numpy', {'I': np.complex128(1j)}])(1 + 1j)
            else:
                sp.sympify(self.expr.split('=')[1] if '=' in self.expr else self.expr)
            if robust_mode:
                expr = sp.sympify(self.expr.split('=')[1] if '=' in self.expr else self.expr)
                if self.plot_type != 'complex' and len(expr.free_symbols) > 2:
                    return False, "Too many free symbols. Use x, y, t, theta, or z only."
                if self.plot_type == 'function' and expr.has(sp.I) and not self.plot_type == 'complex':
                    return False, "Complex numbers not supported in function plots. Try complex plot type."
            return True, ""
        except Exception as e:
            logging.error(f"Validation failed for {self.expr}: {str(e)}")
            return False, f"Invalid expression: {str(e)}. Check syntax (e.g., use * for multiplication)."

class Plotter:
    def __init__(self, ax, x_range, y_range, robust_mode=False, resolution=1000):
        self.ax = ax
        self.x_range = x_range
        self.y_range = y_range
        self.custom_funcs = {}
        self.bg_color = 'white'
        self.grid_style = 'solid'
        self.robust_mode = robust_mode
        self.title = ""
        self.xlabel = "x"
        self.ylabel = "y"
        self.resolution = resolution
        self.grid_intervals = {'major_x': 1, 'major_y': 1, 'minor_x': 0.2, 'minor_y': 0.2}

    def set_background(self, color):
        self.bg_color = color
        self.ax.set_facecolor(color)

    def set_grid_style(self, style):
        self.grid_style = style

    def set_labels(self, title, xlabel, ylabel):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def set_resolution(self, resolution):
        self.resolution = resolution

    def set_grid_intervals(self, major_x, major_y, minor_x, minor_y):
        self.grid_intervals = {'major_x': major_x, 'major_y': major_y, 'minor_x': minor_x, 'minor_y': minor_y}

    def fit_polynomial(self, x_data, y_data, degree):
        coeffs, _ = curve_fit(lambda x, *p: sum(p[i]*x**i for i in range(degree+1)), x_data, y_data, p0=[0]*(degree+1))
        return lambda x: sum(coeffs[i]*x**i for i in range(degree+1))

    def plot(self, expressions, sliders, log_scale=False, show_intersections=False, shade_area=False, show_legend=False, animate=False):
        self.ax.clear()
        self.ax.set_facecolor(self.bg_color)
        self.ax.grid(True, which='major', linestyle=self.grid_style, linewidth=1)
        self.ax.grid(True, which='minor', linestyle=':', linewidth=0.5)
        self.ax.set_xlim(self.x_range)
        self.ax.set_ylim(self.y_range)
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.xaxis.set_major_locator(plt.MultipleLocator(self.grid_intervals['major_x']))
        self.ax.yaxis.set_major_locator(plt.MultipleLocator(self.grid_intervals['major_y']))
        self.ax.xaxis.set_minor_locator(plt.MultipleLocator(self.grid_intervals['minor_x']))
        self.ax.yaxis.set_minor_locator(plt.MultipleLocator(self.grid_intervals['minor_y']))

        # Handle log scale safely
        if log_scale:
            try:
                if self.x_range[0] <= 0 or self.x_range[1] <= 0:
                    raise ValueError("X range must be positive for log scale.")
                if self.y_range[0] <= 0 or self.y_range[1] <= 0:
                    raise ValueError("Y range must be positive for log scale.")
                self.ax.set_xscale('log')
                self.ax.set_yscale('log')
            except ValueError as e:
                logging.error(f"Log scale error: {str(e)}")
                messagebox.showerror("Error", str(e))
                self.ax.set_xscale('linear')
                self.ax.set_yscale('linear')
                log_scale = False  # Fallback to linear scale

        x, y, t, theta, z, I = sp.symbols('x y t theta z I')
        resolution = self.resolution // 2 if self.robust_mode else self.resolution
        x_vals = np.linspace(self.x_range[0], self.x_range[1], resolution)
        t_vals = np.linspace(-10, 10, resolution)
        theta_vals = np.linspace(0, 2*np.pi, resolution)
        time_vals = np.linspace(0, 10, 100) if animate else [0]

        legend_labels = []
        points_plotted = 0
        i = 0
        while i < len(expressions):
            expr = expressions[i]
            if not expr.enabled:
                i += 1
                continue
            try:
                subs = {v: sliders[v][0].get() for v in sliders}
                label = expr.expr if not expr.note else f"{expr.expr} ({expr.note})"
                for time in time_vals if animate else [None]:
                    if expr.plot_type == 'point':
                        x_data, y_data = eval(expr.expr, {'np': np})
                        points_plotted += len(x_data)
                        if expr.error_data:
                            self.ax.errorbar(x_data, y_data, yerr=expr.error_data, fmt='o', color=expr.color, alpha=expr.opacity, label=label)
                        else:
                            self.ax.plot(x_data, y_data, 'o', color=expr.color, alpha=expr.opacity, label=label)
                    elif expr.plot_type == 'polar':
                        if not expr.cached_func:
                            r = sp.lambdify(theta, sp.simplify(sp.sympify(expr.expr.split('=')[1])).subs(subs), 'numpy')
                            expr.cached_func = r
                        r_vals = expr.cached_func(theta_vals + (time or 0))
                        x_vals_polar = r_vals * np.cos(theta_vals)
                        y_vals_polar = r_vals * np.sin(theta_vals)
                        self.ax.plot(x_vals_polar, y_vals_polar, color=expr.color, linestyle=expr.style, alpha=expr.opacity, label=label)
                        points_plotted += len(x_vals_polar)
                    elif expr.plot_type == 'parametric':
                        if i + 1 < len(expressions):
                            x_expr = sp.sympify(expr.expr.split('=')[1]).subs({'t': t + (time or 0)})
                            y_expr = sp.sympify(expressions[i+1].expr.split('=')[1]).subs({'t': t + (time or 0)})
                            x_func = sp.lambdify(t, x_expr.subs(subs), 'numpy')
                            y_func = sp.lambdify(t, y_expr.subs(subs), 'numpy')
                            x_vals_param = x_func(t_vals)
                            y_vals_param = y_func(t_vals)
                            if self.robust_mode:
                                x_vals_param = np.nan_to_num(x_vals_param, nan=0, posinf=1e10, neginf=-1e10)
                                y_vals_param = np.nan_to_num(y_vals_param, nan=0, posinf=1e10, neginf=-1e10)
                            self.ax.plot(x_vals_param, y_vals_param, color=expr.color, linestyle=expr.style, alpha=expr.opacity, label=label)
                            points_plotted += len(t_vals)
                            i += 1
                    elif expr.plot_type == 'implicit' or expr.plot_type == 'conic':
                        X, Y = np.meshgrid(x_vals, np.linspace(self.y_range[0], self.y_range[1], resolution))
                        F = sp.lambdify((x, y), sp.sympify(expr.expr.replace('=', '-')).subs(subs), 'numpy')
                        self.ax.contour(X, Y, F(X, Y), [0], colors=expr.color, linestyles=expr.style, alpha=expr.opacity)
                        points_plotted += resolution * resolution
                    elif expr.plot_type == 'slope_field':
                        X, Y = np.meshgrid(np.linspace(self.x_range[0], self.x_range[1], 20), np.linspace(self.y_range[0], self.y_range[1], 20))
                        dy_dx = sp.lambdify((x, y), sp.sympify(expr.expr.split('=')[1]), 'numpy')
                        U = 1
                        V = dy_dx(X, Y)
                        self.ax.quiver(X, Y, U/np.sqrt(U**2+V**2), V/np.sqrt(U**2+V**2), color=expr.color, alpha=expr.opacity)
                    elif expr.plot_type == 'piecewise':
                        parts = expr.expr.strip('{}').split(',')
                        x_vals_piece = []
                        y_vals_piece = []
                        for part in parts:
                            condition, value = part.split(':')
                            cond_func = sp.lambdify(x, sp.sympify(condition), 'numpy')
                            val_func = sp.lambdify(x, sp.sympify(value).subs(subs), 'numpy')
                            mask = cond_func(x_vals)
                            x_vals_piece.extend(x_vals[mask])
                            y_vals_piece.extend(val_func(x_vals[mask]))
                        self.ax.plot(x_vals_piece, y_vals_piece, color=expr.color, linestyle=expr.style, alpha=expr.opacity, label=label)
                        points_plotted += len(x_vals_piece)
                    elif expr.plot_type in ['derivative', 'integral']:
                        base_expr = sp.sympify(expr.expr.split('=')[1])
                        derived_expr = base_expr.diff(x) if expr.plot_type == 'derivative' else base_expr.integrate(x)
                        if not expr.cached_func:
                            func = sp.lambdify(x, sp.simplify(derived_expr).subs(subs), 'numpy')
                            expr.cached_func = func
                        y_vals = expr.cached_func(x_vals)
                        if self.robust_mode and (np.any(np.isnan(y_vals)) or np.any(np.isinf(y_vals))):
                            y_vals = np.nan_to_num(y_vals, nan=0, posinf=1e10, neginf=-1e10)
                        self.ax.plot(x_vals, y_vals, color=expr.color, linestyle=expr.style, alpha=expr.opacity, label=label)
                        points_plotted += len(x_vals)
                    elif expr.plot_type == 'complex':
                        X, Y = np.meshgrid(x_vals, np.linspace(self.y_range[0], self.y_range[1], resolution))
                        Z = X + 1j * Y
                        if not expr.cached_func:
                            func = sp.lambdify(z, sp.sympify(expr.expr.split('=')[1]).subs(subs), 
                                             modules=['numpy', {'I': np.complex128(1j)}])
                            expr.cached_func = func
                        W = func(Z)
                        # Plot real part by default; could extend to imaginary or magnitude
                        real_vals = np.real(W)
                        if self.robust_mode:
                            real_vals = np.nan_to_num(real_vals, nan=0, posinf=1e10, neginf=-1e10)
                        self.ax.contourf(X, Y, real_vals, levels=20, cmap='viridis', alpha=expr.opacity)
                        self.ax.contour(X, Y, real_vals, levels=10, colors=expr.color, linestyles=expr.style, alpha=expr.opacity)
                        points_plotted += resolution * resolution
                    else:
                        if not expr.cached_func:
                            simplified_expr = sp.simplify(sp.sympify(expr.expr.split('=')[1]).subs(subs))
                            func = sp.lambdify(x, simplified_expr.subs(self.custom_funcs), 'numpy')
                            expr.cached_func = func
                        y_vals = expr.cached_func(x_vals)
                        if self.robust_mode and (np.any(np.isnan(y_vals)) or np.any(np.isinf(y_vals))):
                            logging.warning(f"NaN or Inf detected in {expr.expr}, using fallback rendering")
                            y_vals = np.nan_to_num(y_vals, nan=0, posinf=1e10, neginf=-1e10)
                        self.ax.plot(x_vals, y_vals, color=expr.color, linestyle=expr.style, alpha=expr.opacity, label=label)
                        points_plotted += len(x_vals)
                        if show_intersections and i+1 < len(expressions) and expressions[i+1].plot_type == 'function':
                            next_func = sp.lambdify(x, sp.sympify(expressions[i+1].expr.split('=')[1]).subs(subs), 'numpy')
                            diff = sp.sympify(expr.expr.split('=')[1]) - sp.sympify(expressions[i+1].expr.split('=')[1])
                            roots = sp.solve(diff, x)
                            for r in roots:
                                if r.is_real and self.x_range[0] <= float(r) <= self.x_range[1]:
                                    y_val = expr.cached_func(float(r))
                                    self.ax.plot(float(r), y_val, 'ro')
                        if shade_area and i+1 < len(expressions) and expressions[i+1].plot_type == 'function':
                            next_func = sp.lambdify(x, sp.sympify(expressions[i+1].expr.split('=')[1]).subs(subs), 'numpy')
                            y_vals_next = next_func(x_vals)
                            self.ax.fill_between(x_vals, y_vals, y_vals_next, color=expr.color, alpha=0.3)
                    if animate:
                        plt.pause(0.05)
                if show_legend and expr.enabled and expr.plot_type not in ['implicit', 'slope_field', 'conic', 'complex']:
                    legend_labels.append(label)
            except Exception as e:
                logging.error(f"Plotting failed for {expr.expr}: {str(e)}")
                if self.robust_mode:
                    messagebox.showwarning("Plot Error", f"Failed to plot {expr.expr}: {str(e)}. Try simplifying the expression.")
            i += 1
        if show_legend and legend_labels:
            self.ax.legend(legend_labels, loc='best')
        return points_plotted

class DesmosLikeGraphTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Desmos-Like Graph Tool")
        self.root.resizable(True, True)
        self.expressions = []
        self.undo_stack = []
        self.redo_stack = []
        self.sliders = {}
        self.x_range = (-10, 10)
        self.y_range = (-10, 10)
        self.zoom_factor = 1.0
        self.log_scale = tk.BooleanVar(value=False)
        self.show_intersections = tk.BooleanVar(value=False)
        self.shade_area = tk.BooleanVar(value=False)
        self.compact_mode = tk.BooleanVar(value=False)
        self.robust_mode = tk.BooleanVar(value=False)
        self.show_legend = tk.BooleanVar(value=True)
        self.animate = tk.BooleanVar(value=False)
        self.theme = 'light'
        self.recent_files = []
        self.expression_history = []
        self.auto_save_file = "autosave.json"
        self.sidebar_visible = tk.BooleanVar(value=True)
        self.resolution = 1000
        self.pinned_groups = set()
        self.floating_panel = None
        self.floating_panel_x = 100
        self.floating_panel_y = 100

        # UI Setup
        self.style = ttk.Style()
        self.style.configure('TButton', padding=3, relief='flat', background='#e0e0e0')
        self.style.configure('TLabel', padding=3)
        self.style.configure('TFrame', background='#f5f5f5')
        self.style.map('TButton', background=[('active', '#d0d0d0')])

        # Main Container
        self.main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_frame.pack(fill='both', expand=True)

        # Sidebar (Resizable)
        self.sidebar = ttk.Frame(self.main_frame)
        self.main_frame.add(self.sidebar, weight=1)
        
        # Sidebar Toggle
        self.toggle_button = ttk.Button(self.sidebar, text="◄", command=self.toggle_sidebar, width=2)
        self.toggle_button.pack(side='top', fill='x')

        # Notebook
        self.notebook = ttk.Notebook(self.sidebar)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Expressions Tab
        self.expr_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.expr_tab, text="Expressions")
        self.expr_frame = ttk.LabelFrame(self.expr_tab, text="Add Expression")
        self.expr_frame.pack(fill='x', padx=5, pady=5)
        
        # Expression Entry with Validation
        self.expr_entry = ttk.Entry(self.expr_frame, validate='key', validatecommand=(self.root.register(self.validate_entry), '%P'))
        self.expr_entry.bind('<Return>', lambda e: self.add_expression())
        self.expr_entry.bind('<Up>', lambda e: self.cycle_expression_history(-1))
        self.expr_entry.bind('<Down>', lambda e: self.cycle_expression_history(1))
        self.expr_entry.pack(fill='x', padx=5, pady=5)
        self.expr_entry.configure(foreground='#333333')
        self.validation_label = ttk.Label(self.expr_frame, text="", foreground='red')
        self.validation_label.pack()
        
        # Search
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_expressions)
        ttk.Entry(self.expr_frame, textvariable=self.search_var, width=20).pack(fill='x', padx=5, pady=2)
        
        self.plot_type_var = tk.StringVar(value='function')
        plot_types = ['function', 'parametric', 'polar', 'point', 'implicit', 'slope_field', 'piecewise', 'derivative', 'integral', 'conic', 'complex']
        ttk.OptionMenu(self.expr_frame, self.plot_type_var, 'function', *plot_types).pack(pady=5)
        self.random_color_var = tk.BooleanVar(value=False)
        ttk.Button(self.expr_frame, text="Add", command=self.add_expression).pack(side='left', padx=5)
        ttk.Button(self.expr_frame, text="Templates", command=self.show_templates).pack(side='left', padx=5)
        ttk.Checkbutton(self.expr_frame, text="Random Color", variable=self.random_color_var).pack(side='left', padx=5)

        self.expr_listbox = tk.Listbox(self.expr_tab, height=10, width=40, selectmode=tk.SINGLE, exportselection=0, bg='#ffffff', fg='#333333')
        self.expr_listbox.pack(fill='both', padx=5, pady=5, expand=True)
        self.expr_listbox.bind('<Button-3>', self.show_context_menu)
        self.expr_listbox.bind('<Double-1>', self.toggle_expression_visibility)
        self.expr_listbox.bind('<B1-Motion>', self.drag_drop)

        # Controls Tab
        self.control_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.control_tab, text="Controls")
        self.slider_frame = ttk.LabelFrame(self.control_tab, text="Sliders & Range")
        self.slider_frame.pack(fill='both', padx=5, pady=5, expand=True)
        self.range_frame = ttk.Frame(self.slider_frame)
        self.range_frame.pack(fill='x', padx=5)
        ttk.Label(self.range_frame, text="X:").pack(side='left')
        self.x_min_entry = ttk.Entry(self.range_frame, width=5)
        self.x_min_entry.insert(0, "-10")
        self.x_max_entry = ttk.Entry(self.range_frame, width=5)
        self.x_max_entry.insert(0, "10")
        self.x_min_entry.pack(side='left')
        self.x_max_entry.pack(side='left')
        ttk.Label(self.range_frame, text="Y:").pack(side='left')
        self.y_min_entry = ttk.Entry(self.range_frame, width=5)
        self.y_min_entry.insert(0, "-10")
        self.y_max_entry = ttk.Entry(self.range_frame, width=5)
        self.y_max_entry.insert(0, "10")
        self.y_min_entry.pack(side='left')
        self.y_max_entry.pack(side='left')
        ttk.Button(self.range_frame, text="Update", command=self.update_range).pack(side='left', padx=5)
        ttk.Checkbutton(self.slider_frame, text="Log Scale", variable=self.log_scale, command=self.plot_graph).pack(anchor='w')
        ttk.Checkbutton(self.slider_frame, text="Show Intersections", variable=self.show_intersections, command=self.plot_graph).pack(anchor='w')
        ttk.Checkbutton(self.slider_frame, text="Shade Area", variable=self.shade_area, command=self.plot_graph).pack(anchor='w')
        ttk.Checkbutton(self.slider_frame, text="Show Legend", variable=self.show_legend, command=self.plot_graph).pack(anchor='w')
        ttk.Checkbutton(self.slider_frame, text="Animate", variable=self.animate, command=self.plot_graph).pack(anchor='w')

        # Settings Tab
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="Settings")
        ttk.Checkbutton(self.settings_tab, text="Compact Mode", variable=self.compact_mode, command=self.toggle_compact).pack(pady=5, anchor='w')
        ttk.Checkbutton(self.settings_tab, text="Robust Mode", variable=self.robust_mode, command=self.plot_graph).pack(pady=5, anchor='w')
        ttk.Label(self.settings_tab, text="Theme:").pack(anchor='w')
        self.theme_var = tk.StringVar(value='light')
        ttk.OptionMenu(self.settings_tab, self.theme_var, 'light', 'light', 'dark', 'high-contrast', command=self.update_theme).pack(anchor='w')
        ttk.Button(self.settings_tab, text="Set Background Color", command=self.set_background_color).pack(pady=5, anchor='w')
        ttk.Label(self.settings_tab, text="Grid Style:").pack(anchor='w')
        self.grid_style_var = tk.StringVar(value='solid')
        ttk.OptionMenu(self.settings_tab, self.grid_style_var, 'solid', 'solid', 'dashed', 'dotted', command=self.update_grid_style).pack(anchor='w')
        ttk.Button(self.settings_tab, text="Set Grid Intervals", command=self.set_grid_intervals).pack(pady=5, anchor='w')
        ttk.Button(self.settings_tab, text="Set Plot Labels", command=self.set_plot_labels).pack(pady=5, anchor='w')
        ttk.Label(self.settings_tab, text="Plot Resolution:").pack(anchor='w')
        self.resolution_var = tk.StringVar(value='1000')
        ttk.OptionMenu(self.settings_tab, self.resolution_var, '1000', '500', '1000', '2000', command=self.update_resolution).pack(anchor='w')
        ttk.Button(self.settings_tab, text="Export PDF", command=self.export_pdf).pack(pady=5, anchor='w')

        # Plot Area
        self.plot_frame = ttk.Frame(self.main_frame)
        self.main_frame.add(self.plot_frame, weight=3)

        # Plot Canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.plotter = Plotter(self.ax, self.x_range, self.y_range, self.robust_mode.get(), self.resolution)
        self.canvas.mpl_connect('motion_notify_event', self.update_status)
        self.canvas.mpl_connect('button_press_event', self.handle_canvas_click)
        self.canvas.mpl_connect('motion_notify_event', self.pan)
        self.canvas.mpl_connect('button_release_event', self.end_pan)
        self.pan_start = None

        # Menubar
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save JSON", command=self.save_json, accelerator="Ctrl+S")
        file_menu.add_command(label="Load JSON", command=self.load_json, accelerator="Ctrl+O")
        self.recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Files", menu=self.recent_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Export PNG", command=self.export_png)
        file_menu.add_command(label="Export SVG", command=self.save_svg)
        file_menu.add_command(label="Export PDF", command=self.export_pdf)
        file_menu.add_command(label="Export CSV", command=self.export_csv)

        edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_command(label="Clear All", command=self.clear_all)
        edit_menu.add_command(label="Fit Data", command=self.fit_data)

        view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Sidebar", variable=self.sidebar_visible, command=self.toggle_sidebar)
        view_menu.add_checkbutton(label="Compact Mode", variable=self.compact_mode, command=self.toggle_compact)

        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)

        # Toolbar (Normal Mode)
        self.toolbar = ttk.Frame(self.plot_frame, style='TFrame')
        self.toolbar.pack(fill='x', pady=2)
        self.create_toolbar_button("Clear", self.clear_all, "Clear all expressions", "🗑")
        self.create_toolbar_button("Undo", self.undo, "Undo last action (Ctrl+Z)", "↺")
        self.create_toolbar_button("Redo", self.redo, "Redo last action (Ctrl+Y)", "↻")
        self.grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.toolbar, text="Grid", variable=self.grid_var, command=self.plot_graph).pack(side='left', padx=2)
        self.create_toolbar_button("Theme", self.toggle_theme, "Toggle light/dark theme", "🎨")
        self.create_toolbar_button("Save", self.save_json, "Save to JSON (Ctrl+S)", "💾")
        self.create_toolbar_button("Load", self.load_json, "Load from JSON (Ctrl+O)", "📂")
        self.create_toolbar_button("SVG", self.save_svg, "Export as SVG", "🖼")
        self.create_toolbar_button("PDF", self.export_pdf, "Export as PDF", "📄")
        self.create_toolbar_button("CSV", self.export_csv, "Export as CSV", "📊")
        self.create_toolbar_button("Batch", self.batch_import, "Batch import from CSV/JSON", "📑")
        self.create_toolbar_button("Annotate", self.add_annotation, "Add annotation to plot", "✍")
        self.create_toolbar_button("Zoom In", self.zoom_in, "Zoom in", "🔎+")
        self.create_toolbar_button("Zoom Out", self.zoom_out, "Zoom out", "🔎-")
        self.create_toolbar_button("Fit", self.fit_data, "Fit curve to point data", "📈")

        # Compact Mode Toolbar
        self.compact_toolbar = None

        # Status Bar
        self.status_var = tk.StringVar(value="Zoom: 1.0x | Mouse: (0, 0) | y=N/A | Expressions: 0 | Points: 0")
        self.status_bar = ttk.Label(self.plot_frame, textvariable=self.status_var, anchor='w', relief='sunken', style='TFrame')
        self.status_bar.pack(fill='x', pady=2)

        # Context Menu
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="Copy", command=self.copy_expression)
        self.context_menu.add_command(label="Edit", command=self.edit_expression)
        self.context_menu.add_command(label="Delete", command=self.delete_expression)
        self.context_menu.add_command(label="Group", command=self.set_group)
        self.context_menu.add_command(label="Pin Group", command=self.pin_group)
        self.context_menu.add_command(label="Plot Options", command=self.show_plot_options)
        self.context_menu.add_command(label="Add Note", command=self.add_note)
        self.context_menu.add_command(label="Add Error Bars", command=self.add_error_bars)

        # Keyboard Shortcuts
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Control-y>', lambda e: self.redo())
        self.root.bind('<Control-s>', lambda e: self.save_json())
        self.root.bind('<Control-o>', lambda e: self.load_json())
        self.root.bind('<Control-f>', lambda e: self.expr_entry.focus_set() if not self.compact_mode.get() else self.toggle_floating_panel())
        self.root.bind('<Control-n>', lambda e: self.add_expression())
        self.root.bind('<Tab>', lambda e: self.notebook.focus_set() if not self.compact_mode.get() else None)
        self.root.bind('<Control-t>', lambda e: self.toggle_sidebar() if not self.compact_mode.get() else None)
        self.root.bind('<Control-l>', lambda e: self.set_plot_labels())
        self.root.bind('<Control-r>', lambda e: self.plot_graph())
        self.root.bind('<Control-g>', lambda e: self.toggle_group_visibility())

        # Accessibility
        self.expr_entry.configure(takefocus=True)
        self.expr_listbox.configure(takefocus=True)
        self.canvas.get_tk_widget().configure(takefocus=True)
        self.root.option_add('*TButton*highlightThickness', 2)
        self.root.option_add('*TEntry*highlightThickness', 2)

        # Auto-save Setup
        self.root.after(30000, self.auto_save)

        self.update_recent_menu()
        self.update_theme('light')
        self.plot_graph()

    def create_toolbar_button(self, text, command, tooltip, icon):
        btn = ttk.Button(self.toolbar, text=f"{icon} {text}", command=command, width=8)
        btn.pack(side='left', padx=2)
        btn.bind('<Enter>', lambda e: self.status_var.set(tooltip + " (e.g., y=x^2 for functions, r=cos(theta) for polar)"))
        btn.bind('<Leave>', lambda e: self.update_status(None))
        btn.configure(takefocus=True)

    def create_compact_toolbar(self):
        self.compact_toolbar = ttk.Frame(self.plot_frame, style='TFrame')
        self.compact_toolbar.place(relx=0.5, rely=0, anchor='n')
        self.compact_toolbar.bind('<Enter>', lambda e: self.compact_toolbar.lift())
        self.compact_toolbar.bind('<Leave>', lambda e: self.compact_toolbar.lower() if not self.floating_panel else None)
        buttons = [
            ("Expr", self.toggle_floating_panel, "Show/hide expression panel (Ctrl+F)", "📝"),
            ("Zoom In", self.zoom_in, "Zoom in", "🔎+"),
            ("Zoom Out", self.zoom_out, "Zoom out", "🔎-"),
            ("Labels", self.set_plot_labels, "Set plot labels (Ctrl+L)", "🏷"),
            ("Redraw", self.plot_graph, "Redraw plot (Ctrl+R)", "🔄"),
            ("Fit", self.fit_data, "Fit curve to point data", "📈")
        ]
        for text, cmd, tip, icon in buttons:
            btn = ttk.Button(self.compact_toolbar, text=f"{icon} {text}", command=cmd, width=8)
            btn.pack(side='left', padx=2)
            btn.bind('<Enter>', lambda e, t=tip: self.status_var.set(t))
            btn.bind('<Leave>', lambda e: self.update_status(None))
            btn.configure(takefocus=True)

    def validate_entry(self, text):
        if not text:
            self.validation_label.config(text="")
            return True
        expr = Expression(text, self.plot_type_var.get())
        valid, msg = expr.validate(self.robust_mode.get())
        self.validation_label.config(text=msg if not valid else "Valid", foreground='red' if not valid else 'green')
        return True

    def toggle_sidebar(self):
        if self.sidebar_visible.get():
            self.main_frame.forget(self.sidebar)
            self.toggle_button.config(text="►")
        else:
            self.main_frame.add(self.sidebar, weight=1)
            self.toggle_button.config(text="◄")
        self.sidebar_visible.set(not self.sidebar_visible.get())

    def toggle_compact(self):
        if self.compact_mode.get():
            self.main_frame.pack_forget()
            self.toolbar.pack_forget()
            self.status_bar.pack_forget()
            self.plot_frame.pack(fill='both', expand=True)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
            self.create_compact_toolbar()
            self.compact_toolbar.pack(fill='x')
            self.toggle_floating_panel()
        else:
            if self.floating_panel:
                self.floating_panel.destroy()
                self.floating_panel = None
            if self.compact_toolbar:
                self.compact_toolbar.destroy()
                self.compact_toolbar = None
            self.plot_frame.pack_forget()
            self.main_frame.pack(fill='both', expand=True)
            self.toolbar.pack(fill='x', pady=2)
            self.plot_frame.pack(side='left', fill='both', expand=True)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
            self.status_bar.pack(fill='x', pady=2)
            if self.sidebar_visible.get():
                self.main_frame.add(self.sidebar, weight=1)

    def toggle_floating_panel(self):
        if self.floating_panel:
            self.floating_panel.destroy()
            self.floating_panel = None
        else:
            self.floating_panel = tk.Toplevel(self.root)
            self.floating_panel.overrideredirect(True)
            self.floating_panel.geometry(f"300x400+{self.floating_panel_x}+{self.floating_panel_y}")
            self.floating_panel.attributes('-topmost', True)
            
            def start_drag(event):
                self.floating_panel_x = event.x_root - self.floating_panel.winfo_x()
                self.floating_panel_y = event.y_root - self.floating_panel.winfo_y()

            def drag(event):
                x = event.x_root - self.floating_panel_x
                y = event.y_root - self.floating_panel_y
                self.floating_panel.geometry(f"+{x}+{y}")

            self.floating_panel.bind('<Button-1>', start_drag)
            self.floating_panel.bind('<B1-Motion>', drag)

            frame = ttk.Frame(self.floating_panel, style='TFrame')
            frame.pack(fill='both', expand=True, padx=5, pady=5)
            ttk.Button(frame, text="X", command=self.toggle_floating_panel, width=3).pack(anchor='ne')
            entry = ttk.Entry(frame, validate='key', validatecommand=(self.root.register(self.validate_entry), '%P'))
            entry.pack(fill='x')
            entry.bind('<Return>', lambda e: self.add_expression())
            entry.bind('<Up>', lambda e: self.cycle_expression_history(-1))
            entry.bind('<Down>', lambda e: self.cycle_expression_history(1))
            ttk.Label(frame, textvariable=self.validation_label).pack()
            ttk.OptionMenu(frame, self.plot_type_var, 'function', *['function', 'parametric', 'polar', 'point', 'implicit', 'slope_field', 'piecewise', 'derivative', 'integral', 'conic', 'complex']).pack()
            ttk.Button(frame, text="Add", command=self.add_expression).pack()
            listbox = tk.Listbox(frame, height=10, width=40, selectmode=tk.SINGLE, bg='#ffffff', fg='#333333')
            listbox.pack(fill='both', expand=True)
            listbox.bind('<Button-3>', self.show_context_menu)
            listbox.bind('<Double-1>', self.toggle_expression_visibility)
            for expr in self.expressions:
                display = f"{expr.expr} [{expr.group}]" if expr.group else expr.expr
                if expr.note:
                    display += f" ({expr.note})"
                listbox.insert(tk.END, display)
                listbox.itemconfig(tk.END, fg='gray' if not expr.enabled else expr.color)
            self.expr_listbox = listbox

    def add_expression(self):
        expr = self.expr_entry.get().strip()
        if expr:
            self.save_state()
            color = random.choice(['blue', 'red', 'green', 'purple', 'orange']) if self.random_color_var.get() else 'blue'
            if expr.startswith('f(x)='):
                try:
                    x = sp.symbols('x')
                    self.plotter.custom_funcs['f'] = sp.simplify(sp.sympify(expr.split('=')[1]))
                    self.expressions.append(Expression(expr, 'function', color, note="Custom Function"))
                    self.expression_history.append(expr)
                    if len(self.expression_history) > 10:
                        self.expression_history.pop(0)
                    self.update_expression_list()
                    self.expr_entry.delete(0, tk.END)
                    self.plot_graph()
                except Exception as e:
                    logging.error(f"Custom function error: {str(e)}")
                    messagebox.showerror("Error", f"Invalid custom function: {str(e)}. Use explicit multiplication (e.g., 2*x).")
            else:
                new_expr = Expression(expr, self.plot_type_var.get(), color)
                valid, msg = new_expr.validate(self.robust_mode.get())
                if valid:
                    self.expressions.append(new_expr)
                    self.expression_history.append(expr)
                    if len(self.expression_history) > 10:
                        self.expression_history.pop(0)
                    self.update_expression_list()
                    self.expr_entry.delete(0, tk.END)
                    self.update_sliders(expr)
                    self.plot_graph()
                else:
                    messagebox.showerror("Error", msg)
            self.update_status(None)

    def cycle_expression_history(self, direction):
        if not self.expression_history:
            return
        current = self.expr_entry.get()
        try:
            idx = self.expression_history.index(current) if current in self.expression_history else -1
            new_idx = (idx + direction) % len(self.expression_history)
            self.expr_entry.delete(0, tk.END)
            self.expr_entry.insert(0, self.expression_history[new_idx])
        except:
            self.expr_entry.delete(0, tk.END)
            self.expr_entry.insert(0, self.expression_history[-1])

    def update_expression_list(self):
        self.expr_listbox.delete(0, tk.END)
        for expr in self.expressions:
            icon = {
                'function': '📈', 'parametric': '🔄', 'polar': '🌐', 'point': '⚫',
                'implicit': '🟰', 'slope_field': '➶', 'piecewise': '🔗', 
                'derivative': '📉', 'integral': '∫', 'conic': '⚪', 'complex': 'ℂ'
            }[expr.plot_type]
            display = f"{icon} {expr.expr} [{expr.group}]" if expr.group else f"{icon} {expr.expr}"
            if expr.note:
                display += f" ({expr.note})"
            self.expr_listbox.insert(tk.END, display)
            self.expr_listbox.itemconfig(tk.END, fg='gray' if not expr.enabled else expr.color, 
                                       bg='#e0e0e0' if self.expr_listbox.curselection() and self.expr_listbox.curselection()[0] == self.expr_listbox.size()-1 else '#ffffff')

    def filter_expressions(self, *args):
        search = self.search_var.get().lower()
        self.expr_listbox.delete(0, tk.END)
        for expr in self.expressions:
            if search in expr.expr.lower() or search in expr.group.lower() or search in expr.note.lower():
                icon = {
                    'function': '📈', 'parametric': '🔄', 'polar': '🌐', 'point': '⚫',
                    'implicit': '🟰', 'slope_field': '➶', 'piecewise': '🔗', 
                    'derivative': '📉', 'integral': '∫', 'conic': '⚪', 'complex': 'ℂ'
                }[expr.plot_type]
                display = f"{icon} {expr.expr} [{expr.group}]" if expr.group else f"{icon} {expr.expr}"
                if expr.note:
                    display += f" ({expr.note})"
                self.expr_listbox.insert(tk.END, display)
                self.expr_listbox.itemconfig(tk.END, fg='gray' if not expr.enabled else expr.color)

    def toggle_expression_visibility(self, event):
        selection = self.expr_listbox.curselection()
        if selection:
            self.save_state()
            self.expressions[selection[0]].enabled = not self.expressions[selection[0]].enabled
            self.update_expression_list()
            self.plot_graph()

    def delete_expression(self):
        selection = self.expr_listbox.curselection()
        if selection:
            self.save_state()
            self.expressions.pop(selection[0])
            self.update_expression_list()
            self.plot_graph()

    def copy_expression(self):
        selection = self.expr_listbox.curselection()
        if selection:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.expressions[selection[0]].expr)

    def edit_expression(self):
        selection = self.expr_listbox.curselection()
        if selection:
            self.expr_entry.delete(0, tk.END)
            self.expr_entry.insert(0, self.expressions[selection[0]].expr)
            self.delete_expression()

    def set_group(self):
        selection = self.expr_listbox.curselection()
        if selection:
            group = simpledialog.askstring("Group", "Enter group name:") or ""
            self.save_state()
            self.expressions[selection[0]].group = group
            self.update_expression_list()

    def pin_group(self):
        selection = self.expr_listbox.curselection()
        if selection:
            group = self.expressions[selection[0]].group
            if group:
                self.save_state()
                if group in self.pinned_groups:
                    self.pinned_groups.remove(group)
                else:
                    self.pinned_groups.add(group)
                self.update_expression_list()

    def toggle_group_visibility(self):
        selection = self.expr_listbox.curselection()
        if selection:
            group = self.expressions[selection[0]].group
            if group:
                self.save_state()
                enabled = not any(expr.enabled for expr in self.expressions if expr.group == group)
                for expr in self.expressions:
                    if expr.group == group:
                        expr.enabled = enabled
                self.update_expression_list()
                self.plot_graph()

    def add_note(self):
        selection = self.expr_listbox.curselection()
        if selection:
            note = simpledialog.askstring("Note", "Enter note for expression:") or ""
            self.save_state()
            self.expressions[selection[0]].note = note
            self.update_expression_list()

    def add_error_bars(self):
        selection = self.expr_listbox.curselection()
        if selection and self.expressions[selection[0]].plot_type == 'point':
            errors = simpledialog.askstring("Error Bars", "Enter error values (comma-separated or single value):")
            if errors:
                try:
                    x_data, y_data = eval(self.expressions[selection[0]].expr, {'np': np})
                    error_vals = [float(e) for e in errors.split(',')] if ',' in errors else [float(errors)] * len(x_data)
                    self.expressions[selection[0]].error_data = error_vals
                    self.plot_graph()
                except Exception as e:
                    messagebox.showerror("Error", f"Invalid error values: {str(e)}")

    def show_context_menu(self, event):
        self.context_menu.post(event.x_root, event.y_root)

    def show_templates(self):
        win = tk.Toplevel(self.root)
        win.title("Expression Templates")
        templates = [
            'y=x^2', 'y=sin(x)', 'r=cos(3*theta)', 'x^2+y^2=1', 'dy/dx=-x/y',
            'y={x<0:-x,x>=0:x}', 'y=x^3', 'y=exp(x)', 'x^2/4 + y^2/9 = 1',  # Ellipse
            'x^2/4 - y^2/9 = 1',  # Hyperbola
            'y^2 = 4*x',  # Parabola
            'r = theta',  # Spiral
            '(x,y)=(cos(t),sin(t))',  # Parametric circle
            'w=sin(z)'  # Complex function
        ]
        for t in templates:
            ttk.Button(win, text=t, command=lambda e=t: self.load_template(e)).pack(pady=5)

    def load_template(self, expr):
        self.expr_entry.delete(0, tk.END)
        self.expr_entry.insert(0, expr)
        self.plot_type_var.set({
            'x^2+y^2=1': 'conic', 'x^2/4 + y^2/9 = 1': 'conic', 'x^2/4 - y^2/9 = 1': 'conic',
            'y^2 = 4*x': 'conic', 'r = theta': 'polar', '(x,y)=(cos(t),sin(t))': 'parametric',
            'w=sin(z)': 'complex', 'r=cos(3*theta)': 'polar', 'dy/dx=-x/y': 'slope_field'
        }.get(expr, 'function'))
        self.add_expression()

    def show_plot_options(self):
        selection = self.expr_listbox.curselection()
        if selection:
            expr = self.expressions[selection[0]]
            win = tk.Toplevel(self.root)
            win.title("Plot Options")
            ttk.Label(win, text="Color:").pack()
            ttk.Button(win, text="Choose", command=lambda: self.set_color(selection[0])).pack()
            ttk.Label(win, text="Style:").pack()
            style_var = tk.StringVar(value=expr.style)
            ttk.OptionMenu(win, style_var, expr.style, 'solid', 'dashed', 'dotted', command=lambda v: self.set_style(selection[0], v)).pack()
            ttk.Label(win, text="Opacity:").pack()
            opacity_scale = ttk.Scale(win, from_=0, to=1, value=expr.opacity, command=lambda v: self.set_opacity(selection[0], v))
            opacity_scale.pack()
            enabled_var = tk.BooleanVar(value=expr.enabled)
            ttk.Checkbutton(win, text="Enabled", variable=enabled_var, command=lambda: self.toggle_expression(selection[0])).pack()

    def set_color(self, index):
        color = colorchooser.askcolor(title="Choose Color")[1]
        if color:
            self.save_state()
            self.expressions[index].color = color
            self.update_expression_list()
            self.plot_graph()

    def set_style(self, index, style):
        self.save_state()
        self.expressions[index].style = style
        self.plot_graph()

    def set_opacity(self, index, value):
        self.save_state()
        self.expressions[index].opacity = float(value)
        self.plot_graph()

    def toggle_expression(self, index):
        self.save_state()
        self.expressions[index].enabled = not self.expressions[index].enabled
        self.update_expression_list()
        self.plot_graph()

    def update_sliders(self, expr):
        try:
            symbols = sp.sympify(expr.split('=')[1] if '=' in expr else expr).free_symbols
            vars = [str(s) for s in symbols if str(s) not in ['x', 'y', 't', 'theta', 'z', 'I']]
            for var in vars:
                if var not in self.sliders:
                    frame = ttk.Frame(self.slider_frame)
                    frame.pack(fill='x')
                    ttk.Label(frame, text=f"{var}:").pack(side='left')
                    scale = ttk.Scale(frame, from_=-10, to=10, value=1.0, command=lambda v: self.plot_graph())
                    scale.pack(side='left', fill='x', expand=True)
                    value_label = ttk.Label(frame, text="1.0")
                    value_label.pack(side='left')
                    scale.bind('<B1-Motion>', lambda e: value_label.config(text=f"{scale.get():.2f}"))
                    self.sliders[var] = (scale, -10, 10, 1.0)
        except:
            pass

    def plot_graph(self):
        self.plotter.robust_mode = self.robust_mode.get()
        if self.robust_mode.get() and len(self.expressions) > 50:
            if not messagebox.askyesno("Warning", "Large number of expressions may impact performance. Continue?"):
                return
        self.ax.grid(self.grid_var.get())
        self.plotter.x_range = self.x_range
        self.plotter.y_range = self.y_range
        start_time = datetime.now()
        points_plotted = self.plotter.plot(self.expressions, self.sliders, self.log_scale.get(), 
                                          self.show_intersections.get(), self.shade_area.get(), 
                                          self.show_legend.get(), self.animate.get())
        elapsed = (datetime.now() - start_time).total_seconds()
        if self.robust_mode.get() and elapsed > 5:
            logging.warning(f"Plotting took {elapsed:.2f} seconds")
            messagebox.showwarning("Performance", f"Plotting took {elapsed:.2f} seconds. Consider reducing resolution or simplifying expressions.")
        self.canvas.draw()
        self.update_status(None, points_plotted)

    def update_range(self):
        try:
            x_min = float(self.x_min_entry.get())
            x_max = float(self.x_max_entry.get())
            y_min = float(self.y_min_entry.get())
            y_max = float(self.y_max_entry.get())
            if x_min < x_max and y_min < y_max:
                self.save_state()
                self.x_range = (x_min, x_max)
                self.y_range = (y_min, y_max)
                self.plot_graph()
            else:
                messagebox.showerror("Error", "Invalid range: Ensure min < max.")
        except:
            messagebox.showerror("Error", "Enter valid numbers for ranges.")

    def update_resolution(self, resolution):
        self.resolution = int(resolution)
        self.plotter.set_resolution(self.resolution)
        self.plot_graph()

    def set_grid_intervals(self):
        win = tk.Toplevel(self.root)
        win.title("Set Grid Intervals")
        ttk.Label(win, text="Major X Interval:").pack()
        major_x_entry = ttk.Entry(win)
        major_x_entry.insert(0, "1")
        major_x_entry.pack()
        ttk.Label(win, text="Major Y Interval:").pack()
        major_y_entry = ttk.Entry(win)
        major_y_entry.insert(0, "1")
        major_y_entry.pack()
        ttk.Label(win, text="Minor X Interval:").pack()
        minor_x_entry = ttk.Entry(win)
        minor_x_entry.insert(0, "0.2")
        minor_x_entry.pack()
        ttk.Label(win, text="Minor Y Interval:").pack()
        minor_y_entry = ttk.Entry(win)
        minor_y_entry.insert(0, "0.2")
        minor_y_entry.pack()
        ttk.Button(win, text="Apply", command=lambda: self.apply_grid_intervals(
            float(major_x_entry.get()), float(major_y_entry.get()), 
            float(minor_x_entry.get()), float(minor_y_entry.get())
        )).pack(pady=5)

    def apply_grid_intervals(self, major_x, major_y, minor_x, minor_y):
        self.save_state()
        self.plotter.set_grid_intervals(major_x, major_y, minor_x, minor_y)
        self.plot_graph()

    def clear_all(self):
        if messagebox.askyesno("Confirm", "Clear all expressions and settings?"):
            self.save_state()
            self.expressions.clear()
            self.expr_listbox.delete(0, tk.END)
            for widget in self.slider_frame.winfo_children()[1:]:
                widget.destroy()
            self.sliders.clear()
            self.plotter.custom_funcs.clear()
            self.x_range = (-10, 10)
            self.y_range = (-10, 10)
            self.zoom_factor = 1.0
            self.plotter.set_labels("", "x", "y")
            self.plotter.set_grid_intervals(1, 1, 0.2, 0.2)
            self.update_entry_fields()
            self.plot_graph()

    def fit_data(self):
        selection = self.expr_listbox.curselection()
        if selection and self.expressions[selection[0]].plot_type == 'point':
            degree = simpledialog.askinteger("Fit Polynomial", "Enter polynomial degree:", minvalue=1, maxvalue=5)
            if degree:
                try:
                    x_data, y_data = eval(self.expressions[selection[0]].expr, {'np': np})
                    fit_func = self.plotter.fit_polynomial(x_data, y_data, degree)
                    x_vals = np.linspace(min(x_data), max(x_data), 100)
                    y_vals = fit_func(x_vals)
                    fit_expr = f"y={'+'.join(f'{c:.2f}*x^{i}' for i, c in enumerate(fit_func.__code__.co_varnames[1:]))}"
                    self.expressions.append(Expression(fit_expr, 'function', 'red', note=f"Fit (degree {degree})"))
                    self.update_expression_list()
                    self.plot_graph()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to fit data: {str(e)}")

    def handle_canvas_click(self, event):
        if event.button == 1 and event.xdata and event.ydata and self.plot_type_var.get() == 'point':
            self.save_state()
            for i, expr in enumerate(self.expressions):
                if expr.plot_type == 'point' and expr.enabled:
                    x_data, y_data = eval(expr.expr, {'np': np})
                    x_data = list(x_data) + [event.xdata]
                    y_data = list(y_data) + [event.ydata]
                    self.expressions[i].expr = f"[{','.join(f'({x},{y})' for x,y in zip(x_data, y_data))}]"
                    self.update_expression_list()
                    self.plot_graph()
                    return
            self.expressions.append(Expression(f"[({event.xdata},{event.ydata})]", 'point', 'blue'))
            self.update_expression_list()
            self.plot_graph()

    def save_state(self):
        state = {
            'expressions': deepcopy(self.expressions),
            'x_range': self.x_range,
            'y_range': self.y_range,
            'title': self.plotter.title,
            'xlabel': self.plotter.xlabel,
            'ylabel': self.plotter.ylabel,
            'grid_intervals': self.plotter.grid_intervals
        }
        self.undo_stack.append(state)
        self.redo_stack.clear()
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append({
                'expressions': deepcopy(self.expressions),
                'x_range': self.x_range,
                'y_range': self.y_range,
                'title': self.plotter.title,
                'xlabel': self.plotter.xlabel,
                'ylabel': self.plotter.ylabel,
                'grid_intervals': self.plotter.grid_intervals
            })
            state = self.undo_stack.pop()
            self.expressions = state['expressions']
            self.x_range = state['x_range']
            self.y_range = state['y_range']
            self.plotter.set_labels(state['title'], state['xlabel'], state['ylabel'])
            self.plotter.set_grid_intervals(**state['grid_intervals'])
            self.update_expression_list()
            self.update_entry_fields()
            self.plot_graph()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append({
                'expressions': deepcopy(self.expressions),
                'x_range': self.x_range,
                'y_range': self.y_range,
                'title': self.plotter.title,
                'xlabel': self.plotter.xlabel,
                'ylabel': self.plotter.ylabel,
                'grid_intervals': self.plotter.grid_intervals
            })
            state = self.redo_stack.pop()
            self.expressions = state['expressions']
            self.x_range = state['x_range']
            self.y_range = state['y_range']
            self.plotter.set_labels(state['title'], state['xlabel'], state['ylabel'])
            self.plotter.set_grid_intervals(**state['grid_intervals'])
            self.update_expression_list()
            self.update_entry_fields()
            self.plot_graph()

    def update_theme(self, theme):
        self.theme = theme
        if theme == 'light':
            plt.style.use('default')
            self.plotter.set_background('white')
            self.style.configure('TFrame', background='#f5f5f5')
            self.style.configure('TButton', background='#e0e0e0')
            self.expr_listbox.configure(bg='#ffffff', fg='#333333')
        elif theme == 'dark':
            plt.style.use('dark_background')
            self.plotter.set_background('#1f1f1f')
            self.style.configure('TFrame', background='#2f2f2f')
            self.style.configure('TButton', background='#4f4f4f')
            self.expr_listbox.configure(bg='#2f2f2f', fg='#ffffff')
        elif theme == 'high-contrast':
            plt.style.use('default')
            self.plotter.set_background('#ffffff')
            self.style.configure('TFrame', background='#000000')
            self.style.configure('TButton', background='#ffffff', foreground='#000000')
            self.expr_listbox.configure(bg='#000000', fg='#ffffff')
        self.plot_graph()

    def toggle_theme(self):
        themes = ['light', 'dark', 'high-contrast']
        self.theme = themes[(themes.index(self.theme) + 1) % len(themes)]
        self.theme_var.set(self.theme)
        self.update_theme(self.theme)

    def save_json(self):
        file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file:
            data = [{'expr': e.expr, 'plot_type': e.plot_type, 'color': e.color, 'style': e.style, 
                     'enabled': e.enabled, 'note': e.note, 'opacity': e.opacity, 'group': e.group, 
                     'error_data': e.error_data} for e in self.expressions]
            with open(file, 'w') as f:
                json.dump(data, f)
            self.add_recent_file(file)
            messagebox.showinfo("Success", "Expressions saved")

    def load_json(self):
        file = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file:
            self.save_state()
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                self.expressions = [Expression(d['expr'], d['plot_type'], d['color'], d['style'], 
                                            d['enabled'], d['note'], d['group']) for d in data]
                for i, e in enumerate(self.expressions):
                    e.opacity = data[i]['opacity']
                    e.error_data = data[i].get('error_data')
                self.update_expression_list()
                for expr in self.expressions:
                    self.update_sliders(expr.expr)
                self.add_recent_file(file)
                self.plot_graph()
            except Exception as e:
                logging.error(f"JSON load failed: {str(e)}")
                messagebox.showerror("Error", f"Failed to load JSON: {str(e)}. Ensure file is valid JSON.")

    def auto_save(self):
        if self.expressions:
            data = [{'expr': e.expr, 'plot_type': e.plot_type, 'color': e.color, 'style': e.style, 
                     'enabled': e.enabled, 'note': e.note, 'opacity': e.opacity, 'group': e.group, 
                     'error_data': e.error_data} for e in self.expressions]
            try:
                with open(self.auto_save_file, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                logging.error(f"Auto-save failed: {str(e)}")
        self.root.after(30000, self.auto_save)

    def add_recent_file(self, file):
        if file in self.recent_files:
            self.recent_files.remove(file)
        self.recent_files.insert(0, file)
        if len(self.recent_files) > 5:
            self.recent_files.pop()
        self.update_recent_menu()

    def update_recent_menu(self):
        self.recent_menu.delete(0, tk.END)
        for file in self.recent_files:
            if os.path.exists(file):
                self.recent_menu.add_command(label=os.path.basename(file), 
                                           command=lambda f=file: self.load_json_from_recent(f))

    def load_json_from_recent(self, file):
        if os.path.exists(file):
            self.save_state()
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                self.expressions = [Expression(d['expr'], d['plot_type'], d['color'], d['style'], 
                                            d['enabled'], d['note'], d['group']) for d in data]
                for i, e in enumerate(self.expressions):
                    e.opacity = data[i]['opacity']
                    e.error_data = data[i].get('error_data')
                self.update_expression_list()
                for expr in self.expressions:
                    self.update_sliders(expr.expr)
                self.plot_graph()
            except Exception as e:
                logging.error(f"Recent JSON load failed: {str(e)}")
                messagebox.showerror("Error", f"Failed to load JSON: {str(e)}. Ensure file is valid JSON.")

    def save_svg(self):
        file = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG files", "*.svg")])
        if file:
            self.figure.savefig(file, format='svg')
            self.add_recent_file(file)
            messagebox.showinfo("Success", "Plot saved")

    def export_png(self):
        file = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file:
            self.figure.savefig(file, format='png')
            self.add_recent_file(file)
            messagebox.showinfo("Success", "Plot saved")

    def export_pdf(self):
        file = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if file:
            with PdfPages(file) as pdf:
                pdf.savefig(self.figure)
            self.add_recent_file(file)
            messagebox.showinfo("Success", "Plot saved as PDF")

    def export_csv(self):
        file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file:
            x = sp.symbols('x')
            x_vals = np.linspace(self.x_range[0], self.x_range[1], 100)
            data = {'x': x_vals}
            for expr in self.expressions:
                if expr.enabled and expr.plot_type == 'function':
                    try:
                        func = sp.lambdify(x, sp.sympify(expr.expr.split('=')[1]), 'numpy')
                        y_vals = func(x_vals)
                        data[expr.expr] = y_vals
                    except Exception as e:
                        logging.error(f"CSV export failed for {expr.expr}: {str(e)}")
                        continue
            try:
                with open(file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data.keys())
                    for i in range(len(x_vals)):
                        row = [data[key][i] for key in data]
                        writer.writerow(row)
                self.add_recent_file(file)
                messagebox.showinfo("Success", "Data exported to CSV")
            except Exception as e:
                logging.error(f"CSV export failed: {str(e)}")
                messagebox.showerror("Error", f"Unable to export CSV: {str(e)}. Ensure write permissions.")

    def batch_import(self):
        file = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv")])
        if file:
            self.save_state()
            if file.endswith('.json'):
                self.load_json()
            elif file.endswith('.csv'):
                try:
                    with open(file, 'r') as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        x_vals, y_vals = [], []
                        for row in reader:
                            try:
                                x_vals.append(float(row[0]))
                                y_vals.append(float(row[1]))
                            except:
                                continue
                        expr = f"[{','.join(f'({x},{y})' for x,y in zip(x_vals,y_vals))}]"
                        self.expressions.append(Expression(expr, 'point', 'blue'))
                        self.update_expression_list()
                        self.plot_graph()
                    self.add_recent_file(file)
                except Exception as e:
                    logging.error(f"Batch import failed: {str(e)}")
                    messagebox.showerror("Error", f"Failed to import CSV: {str(e)}. Ensure valid CSV format.")

    def add_annotation(self):
        text = simpledialog.askstring("Annotation", "Enter annotation text:")
        if text:
            x = float(simpledialog.askstring("X", "Enter x position:") or 0)
            y = float(simpledialog.askstring("Y", "Enter y position:") or 0)
            self.ax.annotate(text, (x, y), color='black', bbox=dict(facecolor='white', alpha=0.8))
            self.canvas.draw()

    def set_background_color(self):
        color = colorchooser.askcolor(title="Choose Background Color")[1]
        if color:
            self.plotter.set_background(color)
            self.plot_graph()

    def set_plot_labels(self):
        win = tk.Toplevel(self.root)
        win.title("Set Plot Labels")
        ttk.Label(win, text="Title:").pack()
        title_entry = ttk.Entry(win)
        title_entry.pack()
        ttk.Label(win, text="X Label:").pack()
        xlabel_entry = ttk.Entry(win)
        xlabel_entry.pack()
        ttk.Label(win, text="Y Label:").pack()
        ylabel_entry = ttk.Entry(win)
        ylabel_entry.pack()
        ttk.Button(win, text="Apply", command=lambda: self.apply_labels(title_entry.get(), xlabel_entry.get(), ylabel_entry.get())).pack(pady=5)

    def apply_labels(self, title, xlabel, ylabel):
        self.save_state()
        self.plotter.set_labels(title, xlabel, ylabel)
        self.plot_graph()

    def update_grid_style(self, style):
        self.plotter.set_grid_style(style)
        self.plot_graph()

    def zoom_in(self):
        self.zoom_factor *= 1.2
        x_width = (self.x_range[1] - self.x_range[0]) / 1.2
        y_width = (self.y_range[1] - self.y_range[0]) / 1.2
        x_center = (self.x_range[1] + self.x_range[0]) / 2
        y_center = (self.y_range[1] + self.y_range[0]) / 2
        self.x_range = (x_center - x_width/2, x_center + x_width/2)
        self.y_range = (y_center - y_width/2, y_center + y_width/2)
        self.update_entry_fields()
        self.plot_graph()

    def zoom_out(self):
        self.zoom_factor /= 1.2
        x_width = (self.x_range[1] - self.x_range[0]) * 1.2
        y_width = (self.y_range[1] - self.y_range[0]) * 1.2
        x_center = (self.x_range[1] + self.x_range[0]) / 2
        y_center = (self.y_range[1] + self.y_range[0]) / 2
        self.x_range = (x_center - x_width/2, x_center + x_width/2)
        self.y_range = (y_center - y_width/2, y_center + y_width/2)
        self.update_entry_fields()
        self.plot_graph()

    def update_entry_fields(self):
        self.x_min_entry.delete(0, tk.END)
        self.x_max_entry.delete(0, tk.END)
        self.y_min_entry.delete(0, tk.END)
        self.y_max_entry.delete(0, tk.END)
        self.x_min_entry.insert(0, f"{self.x_range[0]:.2f}")
        self.x_max_entry.insert(0, f"{self.x_range[1]:.2f}")
        self.y_min_entry.insert(0, f"{self.y_range[0]:.2f}")
        self.y_max_entry.insert(0, f"{self.y_range[1]:.2f}")

    def update_status(self, event, points_plotted=0):
        status = f"Zoom: {self.zoom_factor:.1f}x | Expressions: {len(self.expressions)} | Points: {points_plotted} | Robust: {'On' if self.robust_mode.get() else 'Off'} | Mode: {'Compact' if self.compact_mode.get() else 'Normal'}"
        if event and event.xdata and event.ydata:
            try:
                x, y = event.xdata, event.ydata
                y_val = "N/A"
                for expr in self.expressions:
                    if expr.enabled and expr.plot_type == 'function':
                        func = sp.lambdify('x', sp.sympify(expr.expr.split('=')[1]), 'numpy')
                        y_val = func(x)
                        break
                status += f" | Mouse: ({x:.2f}, {y:.2f}) | y={y_val:.2f}"
            except:
                status += f" | Mouse: ({event.xdata:.2f}, {event.ydata:.2f}) | y=N/A"
        self.status_var.set(status)


    def pan(self, event):
        if self.pan_start and event.xdata and event.ydata:
            dx = (event.xdata - self.pan_start[1]) * (self.x_range[1] - self.x_range[0]) / self.figure.get_size_inches()[0] / 100
            dy = (event.ydata - self.pan_start[3]) * (self.y_range[1] - self.y_range[0]) / self.figure.get_size_inches()[1] / 100
            self.x_range = (self.x_range[0] - dx, self.x_range[1] - dx)
            self.y_range = (self.y_range[0] - dy, self.y_range[1] - dy)
            self.update_entry_fields()
            self.plot_graph()

    def end_pan(self, event):
        self.pan_start = None

    def drag_drop(self, event):
        pass  # Implement drag-and-drop if needed

    def show_documentation(self):
        win = tk.Toplevel(self.root)
        win.title("Documentation")
        text = tk.Text(win, wrap='word', height=25, width=60)
        text.pack(padx=10, pady=10)
        text.insert(tk.END, """
# 📉 Welcome to **MathTastrophe™**: The Calculator From Your Nightmares

> Because math shouldn't make sense, and neither should the tools you use.

---

## 📛 Name of This Glorious Mistake:
**MathTastrophe™ — Version 0.what.ever**

Because naming it “Desmos” would’ve been illegal, and naming it “Useful” would’ve been a lie.

---

## 📚 What Is This?

A feature-rich (read: bloated), over-designed (read: confusing), and questionably-functioning calculator app with a built-in graphing tool that tries really hard to look smart.

It does **way too much**, but you still won’t understand what’s happening.

---

## 🛠️ Features (That Probably Don't Work Right)

### ✅ Math Stuff You Can Pretend to Understand
- Symbolic math (derivatives, integrals, solving equations)
- Supports complex numbers (like your relationships)
- Piecewise functions (because life is conditional)
- Parametric, polar, and even 3D (which it doesn’t really do well)

### ✅ Graphing
- Click to plot. Click again out of frustration.
- Polar plots (no polar bears were harmed)
- Slope fields (whatever those are)
- Complex graphs (why???)
- Animations (for when still graphs aren't disappointing enough)

### ✅ UI Goodness
- Dark mode (to match your soul)
- Sliders (to look cool while not knowing what you're adjusting)
- Grid styles, zooming, dragging, plot styling — all things you’ll mess up

---

## 🧾 About This Program

**Made by someone who definitely overdid it.**  
What started as a calculator turned into a Graphing Frankenstein™ with more features than friends.

- Codebase is held together with duct tape, `sympy`, and shame.
- Uses `tkinter`, because we like to suffer.
- `matplotlib` for plotting. And rage.
- `pint`, `numpy`, and `scipy` for math it doesn't always get right.

---

## 📦 Requirements

```bash
pip install matplotlib sympy numpy ttkbootstrap pandas seaborn scipy
```

Or just give up and draw graphs by hand.

---

## 🧙 Who Should Use This?

- Students who want to look like they’re trying
- Developers who want to lose hope
- Professors who gave up caring
- Anyone brave enough to type `dy/dx=-x/y` and *hope for a slope field*

---

## 🐞 Bugs?

Yes. Lots. Some even plot.

---

## 🧠 Pro Tip

If something breaks, it’s not a bug — it’s a feature.  
If nothing plots, it's a learning opportunity. For you.

---

## 👀 Final Thoughts

You didn't ask for this.  
You probably don’t need this.  
But here you are, reading this doc like it’ll help.

![why](https://media.giphy.com/media/8L0Pky6C83SzkzU55a/giphy.gif)

Enjoy. Or don’t. We tried. Barely.

        """)
        text.config(state='disabled')

    def show_about(self):
        messagebox.showinfo("About", "Desmos-Like Graph Tool\nVersion 1.3\nCreated with Python, Tkinter, Matplotlib, and SciPy\n© 2025")

if __name__ == "__main__":
    root = tk.Tk()
    app = DesmosLikeGraphTool(root)
    root.mainloop()

    # now just update the code to plot all the type of complex function circle  parabola cuves and etcsac
