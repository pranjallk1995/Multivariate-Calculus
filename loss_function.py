import numpy as np
import config as cfg
import plotly.graph_objs as go

class Loss():

    def __init__(self) -> None:
        self.weight_1 = np.arange(-cfg.GRID_SIZE, cfg.GRID_SIZE, 0.1)
        self.weight_2 = np.copy(self.weight_1)
 
    def loss_function(self) -> np.ndarray:

        # loss = -cos(w_1)-cos(w_2)
        loss_values = np.zeros((self.weight_1.shape[0], self.weight_2.shape[0]))
        for row, weight_1_value in enumerate(self.weight_1):
            for col, weight_2_value in enumerate(self.weight_2):
                loss_values[row][col] = -(np.cos(weight_1_value) + np.cos(weight_2_value))

        return loss_values

class Gradient():

    def __init__(self) -> None:
        pass

    def calulate_gradient(self, input_value_1: np.ndarray, input_value_2: np.ndarray) -> np.ndarray:
        gradient_values = np.zeros((input_value_1.shape[0], input_value_2.shape[0], 2))

        # gradient about input_dimension_1
        # gradient_1(loss) = sin(w_1)
        for row, value_1 in enumerate(input_value_1):
            for col, value_2 in enumerate(input_value_2):
                gradient_values[row][col][0] = np.sin(value_1)

        # gradient about input_dimension_2
        # gradient_2(loss) = sin(w_2)
        for row, value_1 in enumerate(input_value_1):
            for col, value_2 in enumerate(input_value_2):
                gradient_values[row][col][1] = np.sin(value_2)
        
        return gradient_values

class Plot():

    def __init__(self) -> None:
        pass

    def plot_loss_function(self, weight_1: np.ndarray, weight_2: np.ndarray, loss_values: np.ndarray) -> go.Surface:
        return go.Surface(
            x=weight_1, y=weight_2, z=loss_values, colorscale="Teal",
            name="Loss Function", opacity=0.75, showlegend=False, showscale=False
        )
    
    def plot_input_space(self, input_1: np.ndarray, input_2: np.ndarray) -> go.Surface:
        input_space_z_values = np.zeros((input_1.shape[0], input_2.shape[0]))
        return go.Surface(
            x=input_1, y=input_2, z=input_space_z_values, colorscale="Oranges",
            name="Input Space", opacity=0.25, showlegend=False, showscale=False
        )
    
    def plot_gradient_field(self, gradient_values: np.ndarray) -> list:
        gradient_vector_cones = []
        for selected_vector_row in range(gradient_values.shape[0]):
            for selected_vector_col in range(gradient_values.shape[1]):
                if -cfg.VECTOR_SELECTION < gradient_values[selected_vector_row][selected_vector_col][0] < cfg.VECTOR_SELECTION \
                    and -cfg.VECTOR_SELECTION < gradient_values[selected_vector_row][selected_vector_col][1] < cfg.VECTOR_SELECTION:
                    gradient_vector_cones.append(
                        go.Cone(
                            x=[gradient_values[selected_vector_row][selected_vector_col][0]],
                            y=[gradient_values[selected_vector_row][selected_vector_col][1]],
                            z=[0],
                            u=[gradient_values[selected_vector_row][selected_vector_col][0]],
                            v=[gradient_values[selected_vector_row][selected_vector_col][1]],
                            w=[0],
                            showscale=False, showlegend=False, anchor="tip", sizemode="absolute", colorscale="Teal", name="Gradient Direction"
                        )
                    )
        return gradient_vector_cones

    def plot_3d_diagram(self, loss_function: go.Surface, input_space: go.Surface, gradient_field: list) -> None:
        fig = go.Figure([loss_function, input_space] + gradient_field)
        fig.update_layout(
            title="Loss Function", template="plotly_dark",
            scene={
                "xaxis_title": "Weight 1 (w_1)",
                "yaxis_title": "Weight 2 (w_2)",
                "zaxis_title": "Loss = -cos(w_1)-cos(w_2)"
            }
        )
        fig.write_html("loss_function_graph.html")


if __name__ == "__main__":
    
    loss_function_obj = Loss()
    gradient_obj = Gradient()
    plotting_obj = Plot()

    loss_function_plot = plotting_obj.plot_loss_function(
        loss_function_obj.weight_1, loss_function_obj.weight_2, loss_function_obj.loss_function()
    )

    input_space_plot = plotting_obj.plot_input_space(
        loss_function_obj.weight_1, loss_function_obj.weight_2
    )

    gradient_field_plot = plotting_obj.plot_gradient_field(
        gradient_obj.calulate_gradient(loss_function_obj.weight_1, loss_function_obj.weight_2)
    )

    plotting_obj.plot_3d_diagram(
        loss_function_plot, input_space_plot, gradient_field_plot
    )