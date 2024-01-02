import numpy as np
import config as cfg
import streamlit as st
import plotly.graph_objs as go

from plotly.subplots import make_subplots

class LemniscateField():

    def __init__(self) -> None:
        self.x_cords = np.arange(-cfg.X_GRID_SIZE, cfg.X_GRID_SIZE, 0.1)
        self.y_cords = np.arange(-cfg.Y_GRID_SIZE, cfg.Y_GRID_SIZE, 0.1)

    def lemniscate_function(self) -> np.ndarray:
        lemniscate_values = np.zeros((self.x_cords.shape[0], self.y_cords.shape[0]))
        for row, x_cord in enumerate(self.x_cords):
            for col, y_cord in enumerate(self.y_cords):
                lemniscate_values[row][col] = np.power(np.power(x_cord, 2) + np.power(y_cord, 2), 2) \
                    - cfg.C*np.power(x_cord, 2) - cfg.D*np.power(y_cord, 2)
        return lemniscate_values
    
class Gradient():

    def __init__(self) -> None:
        pass

    def calulate_gradient(self, x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        gradient_data = []

        for x_value in x_values:
            for y_value in y_values:
                if round(x_value*10, 0)%2 and round(y_value*10, 0)%2:
                    gradient_data_values = {}
                    gradient_data_values["x_cord"] = x_value
                    gradient_data_values["y_cord"] = y_value
                    gradient_data_values["gradient_x_cord"] = 4*(np.power(x_value, 2) + np.power(y_value, 2))*x_value - 2*cfg.C*x_value
                    gradient_data_values["gradient_y_cord"] = 4*(np.power(x_value, 2) + np.power(y_value, 2))*y_value - 2*cfg.D*y_value
                    gradient_data.append(gradient_data_values)

        return gradient_data

class Plot():

    def __init__(self) -> None:
        pass

    def plot_input_space(self, x_values: np.ndarray, y_values: np.ndarray) -> go.Surface:
        input_space_z_values = cfg.INPUT_SPACE_OFFSET*np.ones((x_values.shape[0], y_values.shape[0]))
        return go.Surface(
            x=x_values, y=y_values, z=input_space_z_values, colorscale="Oranges",
            name="Input Space", opacity=0.25, showlegend=False, showscale=False
        )

    def plot_leminscate_function(self, x_values: np.ndarray, y_values: np.ndarray, lemniscate_values: np.ndarray) -> go.Surface:
        return go.Surface(
            x=x_values, y=y_values, z=lemniscate_values, colorscale="Teal",
            name="Lemniscate Function", opacity=0.75, showlegend=False, showscale=False
        )
    
    def plot_gradient_field(self, gradient_data: list) -> list:
        gradient_vector_cones = []
        for gradient_data_value in gradient_data:
            if -cfg.VECTOR_SELECTION_X < gradient_data_value["x_cord"] < cfg.VECTOR_SELECTION_X \
                and -cfg.VECTOR_SELECTION_Y < gradient_data_value["y_cord"] < cfg.VECTOR_SELECTION_Y:
                gradient_vector_cones.append(
                    go.Cone(
                        x=[gradient_data_value["x_cord"]],
                        y=[gradient_data_value["y_cord"]],
                        z=[cfg.INPUT_SPACE_OFFSET],
                        u=[gradient_data_value["gradient_x_cord"]*cfg.CONE_SCALE],
                        v=[gradient_data_value["gradient_y_cord"]*cfg.CONE_SCALE],
                        w=[0],
                        sizeref=cfg.CONE_SCALE,  sizemode="scaled", showscale=False, showlegend=False,
                        anchor="tip", colorscale="Teal", name="Gradient Direction"
                    )
                )

        return gradient_vector_cones

    def plot_leminscate_contour(self, x_values: np.ndarray, y_values: np.ndarray, lemniscate_values: np.ndarray) -> go.Contour:
        return go.Contour(
            x=x_values, y=y_values, z=lemniscate_values, colorscale="Teal", ncontours=cfg.CONTOURS,
            name="Lemniscate Contour", opacity=0.75, showlegend=False, showscale=False,
            contours={
                "coloring": "heatmap",
                "showlabels": True
            }
        )
    
    def plot_3d_diagram(self, input_space: go.Surface, lemniscate_function: go.Surface, gradient_field: list, lemniscate_contour: go.Contour) -> None:
        fig = make_subplots(
            rows = 1, cols = 2, horizontal_spacing = 0.05, specs=[[{"type": "surface"}, {"type": "contour"}]],
            subplot_titles=("Lemniscate Function", "Leminscate Contour")
        )
        fig.add_trace(lemniscate_function, row=1, col=1)
        fig.add_trace(input_space, row=1, col=1)
        for gradient in gradient_field:
            fig.add_trace(gradient, row=1, col=1)
        fig.add_trace(lemniscate_contour, row=1, col=2)
        fig.update_xaxes(title_text="X-axis", row = 1, col = 2)
        fig.update_yaxes(title_text="Y-axis", row = 1, col = 2)
        fig.update_layout(
            title="Lemniscate Plots", template="plotly_dark",
            scene1={
                "xaxis_title": "X-axis",
                "yaxis_title": "Y-axis",
                "zaxis_title": "(x^2+y^2)^2-16x^2+2y^2"
            }
        )
        fig.write_html("lemniscate_field.html")

if __name__ == "__main__":
    
    lemniscate_obj = LemniscateField()
    gradient_obj = Gradient()
    plotting_obj = Plot()

    input_space_plot = plotting_obj.plot_input_space(
        lemniscate_obj.x_cords, lemniscate_obj.y_cords
    )

    lemniscate_plot = plotting_obj.plot_leminscate_function(
        lemniscate_obj.x_cords, lemniscate_obj.y_cords, lemniscate_obj.lemniscate_function()
    )

    gradient_field_plot = plotting_obj.plot_gradient_field(
        gradient_obj.calulate_gradient(lemniscate_obj.x_cords, lemniscate_obj.y_cords)
    )

    lemniscate_contour_plot = plotting_obj.plot_leminscate_contour(
        lemniscate_obj.x_cords, lemniscate_obj.y_cords, lemniscate_obj.lemniscate_function()
    )

    plotting_obj.plot_3d_diagram(
        input_space_plot, lemniscate_plot, gradient_field_plot, lemniscate_contour_plot
    )
