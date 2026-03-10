import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


class Examples:
    """
    Collection of teaching examples for neural-network notebooks.

    Each method returns either an HTML animation object or a Matplotlib
    figure, depending on the example.
    """

    TITLE_SIZE = 18
    TEXT_SIZE = 14
    PLOT_TEXT_SIZE = 11
    LABEL_SIZE = 16
    TICK_LABEL_SIZE = 14

    @staticmethod
    def activation(z, kind="sigmoid"):
        if kind == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        if kind == "tanh":
            return np.tanh(z)
        if kind == "relu":
            return np.maximum(0.0, z)
        raise ValueError("Unknown activation. Use 'sigmoid', 'tanh', or 'relu'.")

    @staticmethod
    def activation_formula(kind="sigmoid"):
        if kind == "sigmoid":
            return r"$\sigma(z) = \dfrac{1}{1+e^{-z}}$"
        if kind == "tanh":
            return r"$\tanh(z)$"
        if kind == "relu":
            return r"$\mathrm{ReLU}(z) = \max(0, z)$"
        raise ValueError("Unknown activation. Use 'sigmoid', 'tanh', or 'relu'.")

    @classmethod
    def neuron_forward_pass(cls, activation="sigmoid", interval=1500, repeat=True):
        """
        Animated single-neuron forward pass:
        x -> weighted sum z -> activation a = phi(z)
        """
        x = np.array([2.0, -1.5])
        w = np.array([1.2, -0.7])
        b = 0.6

        term1 = w[0] * x[0]
        term2 = w[1] * x[1]
        weighted_sum = term1 + term2
        z = weighted_sum + b
        a = cls.activation(z, activation)

        stages = [
            "Show input vector and parameters",
            "Compute first weighted term",
            "Compute second weighted term",
            "Add weighted terms",
            "Add bias to obtain z",
            "Apply activation to obtain a",
        ]

        fig = plt.figure(figsize=(13, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.15], wspace=0.32)

        ax_text = fig.add_subplot(gs[0, 0])
        ax_plot = fig.add_subplot(gs[0, 1])
        ax_text.axis("off")
        ax_plot.set_box_aspect(1)
        fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.12)

        title_text = ax_text.text(
            0.02, 0.95, "", ha="left", va="top",
            fontsize=cls.TITLE_SIZE, weight="bold", transform=ax_text.transAxes
        )
        line1 = ax_text.text(0.02, 0.80, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line2 = ax_text.text(0.02, 0.68, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line3 = ax_text.text(0.02, 0.56, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line4 = ax_text.text(0.02, 0.42, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line5 = ax_text.text(0.02, 0.28, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line6 = ax_text.text(0.02, 0.14, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)

        z_grid = np.linspace(-5, 5, 400)
        a_grid = cls.activation(z_grid, activation)
        ax_plot.plot(z_grid, a_grid, linewidth=2)
        ax_plot.axhline(0, linewidth=1, color="gray", linestyle="--")
        ax_plot.axvline(0, linewidth=1, color="gray", linestyle="--")
        ax_plot.set_title(f"{activation.capitalize()} activation", fontsize=14)
        ax_plot.set_xlabel(r"$z$")
        ax_plot.set_ylabel(r"$a = \phi(z)$")
        ax_plot.tick_params(labelsize=14)

        moving_point, = ax_plot.plot([], [], "o", markersize=9)
        annotation = ax_plot.text(0.03, 0.95, "", transform=ax_plot.transAxes, va="top", fontsize=14)

        def update(frame):
            stage = stages[frame]
            title_text.set_text(stage)

            line1.set_text(rf"Input vector:  $x = [{x[0]:.2f},\ {x[1]:.2f}]$")
            line2.set_text(rf"Weights:  $w = [{w[0]:.2f},\ {w[1]:.2f}]$,   bias: $b = {b:.2f}$")
            line3.set_text("")
            line4.set_text("")
            line5.set_text("")
            line6.set_text("")
            moving_point.set_data([], [])
            annotation.set_text(cls.activation_formula(activation))

            if frame == 0:
                line3.set_text(r"Neuron equation:  $z = w_1x_1 + w_2x_2 + b$")
                line4.set_text(r"Activation:  $a = \phi(z)$")
                line5.set_text(rf"Chosen activation: {cls.activation_formula(activation)}")
            elif frame == 1:
                line3.set_text(rf"First weighted term:  $w_1x_1 = ({w[0]:.2f})({x[0]:.2f}) = {term1:.2f}$")
                line4.set_text(r"Current computation:  $z = w_1x_1 + w_2x_2 + b$")
            elif frame == 2:
                line3.set_text(rf"Second weighted term:  $w_2x_2 = ({w[1]:.2f})({x[1]:.2f}) = {term2:.2f}$")
                line4.set_text(r"Current computation:  $z = w_1x_1 + w_2x_2 + b$")
            elif frame == 3:
                line3.set_text(rf"Add weighted terms:  ${term1:.2f} + {term2:.2f} = {weighted_sum:.2f}$")
                line4.set_text(rf"Partial result:  $w^\top x = {weighted_sum:.2f}$")
            elif frame == 4:
                line3.set_text(rf"Add bias:  ${weighted_sum:.2f} + {b:.2f} = {z:.2f}$")
                line4.set_text(rf"Pre-activation output:  $z = {z:.2f}$")
                line5.set_text(r"This scalar is the neuron output before nonlinearity.")
            elif frame == 5:
                line3.set_text(rf"Pre-activation value:  $z = {z:.2f}$")
                line4.set_text(r"Apply activation:  $a = \phi(z)$")
                line5.set_text(rf"Activated output:  $a = {a:.4f}$")
                line6.set_text(rf"Final neuron output:  $a = \phi({z:.2f}) = {a:.4f}$")
                moving_point.set_data([z], [a])

            return (
                title_text, line1, line2, line3, line4, line5, line6,
                moving_point, annotation
            )

        anim = FuncAnimation(fig, update, frames=len(stages), interval=interval, blit=False, repeat=repeat)
        plt.close(fig)
        return HTML(anim.to_jshtml())

    @classmethod
    def mse_animation(cls, interval=1800, repeat=True):
        """
        Animated Mean Squared Error example with three samples.
        """
        x = np.array([1, 2, 3], dtype=float)
        y_true = np.array([2, 4, 6], dtype=float)
        y_pred = np.array([1.5, 3.0, 4.5], dtype=float)

        errors = y_pred - y_true
        sq_errors = errors**2
        mse = np.mean(sq_errors)

        stages = [
            "Show true targets",
            "Show model predictions",
            "Draw prediction errors",
            "Square the errors",
            "Average the squared errors",
        ]

        fig = plt.figure(figsize=(13, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.15], wspace=0.32)
        ax_text = fig.add_subplot(gs[0, 0])
        ax_plot = fig.add_subplot(gs[0, 1])
        ax_text.axis("off")
        ax_plot.set_box_aspect(1)
        fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.12)

        title_text = ax_text.text(
            0.02, 0.95, "", ha="left", va="top",
            fontsize=cls.TITLE_SIZE, weight="bold", transform=ax_text.transAxes
        )
        line1 = ax_text.text(0.02, 0.80, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line2 = ax_text.text(0.02, 0.68, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line3 = ax_text.text(0.02, 0.56, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line4 = ax_text.text(0.02, 0.42, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line5 = ax_text.text(0.02, 0.28, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line6 = ax_text.text(0.02, 0.14, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)

        ax_plot.set_title("Predictions, Errors, and MSE", fontsize=cls.TITLE_SIZE, pad=12)
        ax_plot.set_xlabel("Input index", fontsize=cls.LABEL_SIZE)
        ax_plot.set_ylabel("Output value", fontsize=cls.LABEL_SIZE)
        ax_plot.set_xlim(0.5, 3.5)
        ax_plot.set_ylim(0.5, 6.8)
        ax_plot.tick_params(labelsize=cls.TICK_LABEL_SIZE)

        true_scatter = ax_plot.scatter([], [], s=70, label="True targets")
        pred_scatter = ax_plot.scatter([], [], s=70, marker="s", label="Predictions")

        error_lines = []
        error_labels = []
        sq_labels = []
        for _ in range(len(x)):
            line, = ax_plot.plot([], [], linewidth=2, color="red")
            error_lines.append(line)
            err_txt = ax_plot.text(0, 0, "", fontsize=cls.PLOT_TEXT_SIZE)
            sq_txt = ax_plot.text(0, 0, "", fontsize=cls.PLOT_TEXT_SIZE)
            error_labels.append(err_txt)
            sq_labels.append(sq_txt)

        ax_plot.legend(loc="lower right", fontsize=10, frameon=True)

        def reset_artists():
            true_scatter.set_offsets(np.empty((0, 2)))
            pred_scatter.set_offsets(np.empty((0, 2)))
            for line in error_lines:
                line.set_data([], [])
            for txt in error_labels:
                txt.set_text("")
                txt.set_position((0, 0))
            for txt in sq_labels:
                txt.set_text("")
                txt.set_position((0, 0))

        def update(frame):
            title_text.set_text(stages[frame])
            line1.set_text("")
            line2.set_text("")
            line3.set_text("")
            line4.set_text("")
            line5.set_text("")
            line6.set_text("")
            reset_artists()

            true_scatter.set_offsets(np.column_stack([x, y_true]))
            if frame >= 1:
                pred_scatter.set_offsets(np.column_stack([x, y_pred]))
            if frame >= 2:
                for i in range(len(x)):
                    error_lines[i].set_data([x[i], x[i]], [y_true[i], y_pred[i]])
                    mid_y = 0.5 * (y_true[i] + y_pred[i])
                    error_labels[i].set_position((x[i] + 0.07, mid_y))
                    error_labels[i].set_text(rf"$e_{i+1}={errors[i]:.2f}$")
            if frame >= 3:
                for i in range(len(x)):
                    sq_labels[i].set_position((x[i] + 0.07, 0.5 * (y_true[i] + y_pred[i]) + 0.55))
                    sq_labels[i].set_text(rf"$e_{i+1}^2={sq_errors[i]:.2f}$")

            if frame == 0:
                line1.set_text(r"True targets:  $y = [2,\ 4,\ 6]$")
                line2.set_text(r"These points represent the correct outputs.")
                line3.set_text(r"The model will try to predict them as accurately as possible.")
            elif frame == 1:
                line1.set_text(r"Predictions:  $\hat{y} = [1.5,\ 3,\ 4.5]$")
                line2.set_text(r"The model does not match the targets exactly.")
                line3.set_text(r"We now compare predictions and targets sample by sample.")
            elif frame == 2:
                line1.set_text(r"Prediction errors:  $e_i = \hat{y}_i - y_i$")
                line2.set_text(rf"$e = [{errors[0]:.2f},\ {errors[1]:.2f},\ {errors[2]:.2f}]$")
                line3.set_text(r"The vertical segments show the residuals.")
                line4.set_text(r"Negative error means the prediction is too small.")
            elif frame == 3:
                line1.set_text(r"Squared errors:  $\ell_i = (\hat{y}_i - y_i)^2$")
                line2.set_text(rf"$[({errors[0]:.2f})^2,\ ({errors[1]:.2f})^2,\ ({errors[2]:.2f})^2]$")
                line3.set_text(rf"$= [{sq_errors[0]:.2f},\ {sq_errors[1]:.2f},\ {sq_errors[2]:.2f}]$")
                line4.set_text(r"Squaring removes signs and penalizes larger errors.")
            elif frame == 4:
                line1.set_text(r"Mean Squared Error:")
                line2.set_text(
                    rf"$\mathcal{{L}}_{{\mathrm{{MSE}}}}"
                    rf"=\dfrac{{{sq_errors[0]:.2f}+{sq_errors[1]:.2f}+{sq_errors[2]:.2f}}}{{3}}$"
                )
                line3.set_text(rf"$\mathcal{{L}}_{{\mathrm{{MSE}}}} = {mse:.2f}$")
                line4.set_text(r"This single scalar summarizes the error over the dataset.")
                line5.set_text(r"Training aims to reduce this value by changing the parameters.")

            return (
                title_text, line1, line2, line3, line4, line5, line6,
                true_scatter, pred_scatter, *error_lines, *error_labels, *sq_labels
            )

        anim = FuncAnimation(fig, update, frames=len(stages), interval=interval, blit=False, repeat=repeat)
        plt.close(fig)
        return HTML(anim.to_jshtml())

    @classmethod
    def linear_regression_gd_step(cls, eta=0.10, interval=2400, repeat=True):
        """
        Animated linear regression example: loss, gradients, and one gradient-descent step.
        """
        x = np.array([1.0, 2.0, 3.0])
        y_true = np.array([2.0, 4.0, 6.0])
        N = len(x)

        w = 1.5
        b = 0.0

        y_pred = w * x + b
        errors = y_pred - y_true
        sq_errors = errors**2
        mse = np.mean(sq_errors)

        dL_dw_terms = errors * x
        dL_db_terms = errors
        dL_dw = (2 / N) * np.sum(dL_dw_terms)
        dL_db = (2 / N) * np.sum(dL_db_terms)

        w_new = w - eta * dL_dw
        b_new = b - eta * dL_db

        y_pred_new = w_new * x + b_new
        mse_new = np.mean((y_pred_new - y_true) ** 2)

        stages = [
            "Show the dataset",
            "Apply the linear model",
            "Compute the residuals",
            "Compute the Mean Squared Error",
            "Compute the gradient with respect to w",
            "Compute the gradient with respect to b",
            "Take one gradient descent step",
        ]

        fig = plt.figure(figsize=(13, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.15], wspace=0.32)
        ax_text = fig.add_subplot(gs[0, 0])
        ax_plot = fig.add_subplot(gs[0, 1])
        ax_text.axis("off")
        ax_plot.set_box_aspect(1)
        fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.12)

        title_text = ax_text.text(
            0.02, 0.95, "", ha="left", va="top",
            fontsize=cls.TITLE_SIZE, weight="bold", transform=ax_text.transAxes
        )
        line1 = ax_text.text(0.02, 0.80, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line2 = ax_text.text(0.02, 0.68, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line3 = ax_text.text(0.02, 0.56, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line4 = ax_text.text(0.02, 0.42, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line5 = ax_text.text(0.02, 0.28, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)
        line6 = ax_text.text(0.02, 0.14, "", fontsize=cls.TEXT_SIZE, ha="left", transform=ax_text.transAxes)

        ax_plot.set_title("Predictions, loss, and gradients", fontsize=cls.TITLE_SIZE, pad=12)
        ax_plot.set_xlabel("x", fontsize=cls.LABEL_SIZE)
        ax_plot.set_ylabel(r"$y$", fontsize=cls.LABEL_SIZE)
        ax_plot.set_xlim(0.5, 4.0)
        ax_plot.set_ylim(0.5, 6.8)
        ax_plot.tick_params(labelsize=cls.TICK_LABEL_SIZE)

        true_scatter = ax_plot.scatter([], [], s=70, label="True targets")
        x_line = np.linspace(0.5, 3.5, 200)
        y_line = w * x_line + b
        pred_line, = ax_plot.plot([], [], linewidth=2, label=r"Initial model: $\hat{y}=wx+b$")
        pred_scatter = ax_plot.scatter([], [], s=70, marker="s", label="Initial predictions")

        y_line_new = w_new * x_line + b_new
        pred_line_new, = ax_plot.plot([], [], linewidth=2, linestyle="--", label="Updated model")
        pred_scatter_new = ax_plot.scatter([], [], s=70, marker="D", label="Updated predictions")

        error_lines = []
        error_labels = []
        for _ in range(N):
            line, = ax_plot.plot([], [], linewidth=2, color="red")
            txt = ax_plot.text(0, 0, "", fontsize=cls.PLOT_TEXT_SIZE)
            error_lines.append(line)
            error_labels.append(txt)

        ax_plot.legend(loc="lower right", fontsize=9.5, frameon=True)

        def reset_artists():
            true_scatter.set_offsets(np.empty((0, 2)))
            pred_scatter.set_offsets(np.empty((0, 2)))
            pred_scatter_new.set_offsets(np.empty((0, 2)))
            pred_line.set_data([], [])
            pred_line_new.set_data([], [])
            for line in error_lines:
                line.set_data([], [])
            for txt in error_labels:
                txt.set_text("")
                txt.set_position((0, 0))

        def update(frame):
            title_text.set_text(stages[frame])
            line1.set_text("")
            line2.set_text("")
            line3.set_text("")
            line4.set_text("")
            line5.set_text("")
            line6.set_text("")
            reset_artists()

            true_scatter.set_offsets(np.column_stack([x, y_true]))

            if frame >= 1:
                pred_line.set_data(x_line, y_line)
                pred_scatter.set_offsets(np.column_stack([x, y_pred]))

            if 2 <= frame <= 5:
                for i in range(N):
                    error_lines[i].set_data([x[i], x[i]], [y_true[i], y_pred[i]])
                    error_labels[i].set_position((x[i] + 0.08, y_pred[i] - 0.25))
                    error_labels[i].set_text(rf"$e_{i+1}={errors[i]:.2f}$")

            if frame == 6:
                pred_line.set_data(x_line, y_line)
                pred_scatter.set_offsets(np.column_stack([x, y_pred]))
                pred_line_new.set_data(x_line, y_line_new)
                pred_scatter_new.set_offsets(np.column_stack([x, y_pred_new]))

            if frame == 0:
                line1.set_text(r"Dataset:  $x=[1,\ 2,\ 3]$,  $y=[2,\ 4,\ 6]$")
                line2.set_text(r"These blue points are the true targets.")
                line3.set_text(r"We now fit a simple linear model to them.")
            elif frame == 1:
                line1.set_text(r"Model:  $\hat{y}_i = w x_i + b$")
                line2.set_text(rf"Choose parameters:  $w={w:.2f}$,  $b={b:.2f}$")
                line3.set_text(rf"Predictions:  $\hat{{y}}=[{y_pred[0]:.2f},\ {y_pred[1]:.2f},\ {y_pred[2]:.2f}]$")
                line4.set_text(r"The orange squares lie on the prediction line.")
            elif frame == 2:
                line1.set_text(r"Residuals:  $e_i = \hat{y}_i - y_i$")
                line2.set_text(rf"$e=[{errors[0]:.2f},\ {errors[1]:.2f},\ {errors[2]:.2f}]$")
                line3.set_text(r"Each red segment measures prediction minus truth.")
                line4.set_text(r"Negative residuals mean the predictions are too small.")
            elif frame == 3:
                line1.set_text(r"Mean Squared Error:")
                line2.set_text(r"$\mathcal{L}(w,b)=\dfrac{1}{N}\sum_{i=1}^{N}(wx_i+b-y_i)^2$")
                line3.set_text(
                    rf"$\mathcal{{L}}=\dfrac{{{sq_errors[0]:.2f}+{sq_errors[1]:.2f}+{sq_errors[2]:.2f}}}{{3}}={mse:.2f}$"
                )
                line4.set_text(r"This scalar tells us how bad the current parameters are.")
                line5.set_text(r"If we change $w$ or $b$, the predictions and the loss also change.")
            elif frame == 4:
                line1.set_text(r"Gradient with respect to $w$:")
                line2.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial w}=\dfrac{2}{N}\sum_{i=1}^{N}(wx_i+b-y_i)x_i$")
                line3.set_text(
                    rf"$=\dfrac{{2}}{{3}}\left[({errors[0]:.2f})(1)+({errors[1]:.2f})(2)+({errors[2]:.2f})(3)\right]$"
                )
                line4.set_text(
                    rf"$=\dfrac{{2}}{{3}}\left({dL_dw_terms[0]:.2f}+{dL_dw_terms[1]:.2f}+{dL_dw_terms[2]:.2f}\right)$"
                )
                line5.set_text(rf"$\Rightarrow \, \dfrac{{\partial \mathcal{{L}}}}{{\partial w}} = {dL_dw:.2f}$")
                line6.set_text(r"A negative value means increasing $w$ would reduce the loss.")
            elif frame == 5:
                line1.set_text(r"Gradient with respect to $b$:")
                line2.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial b}=\dfrac{2}{N}\sum_{i=1}^{N}(wx_i+b-y_i)$")
                line3.set_text(
                    rf"$=\dfrac{{2}}{{3}}\left({errors[0]:.2f}+{errors[1]:.2f}+{errors[2]:.2f}\right)$"
                )
                line4.set_text(rf"$\Rightarrow \, \dfrac{{\partial \mathcal{{L}}}}{{\partial b}} = {dL_db:.2f}$")
                line5.set_text(r"This tells us how the loss changes if we shift the whole line.")
                line6.set_text(r"These gradients are the quantities used by gradient descent.")
            elif frame == 6:
                line1.set_text(r"One gradient descent step:")
                line2.set_text(rf"$w_{{\mathrm{{new}}}} = w - \eta\, \dfrac{{\partial \mathcal{{L}}}}{{\partial w}} = {w:.2f} - {eta:.2f}({dL_dw:.2f}) = {w_new:.2f}$")
                line3.set_text(rf"$b_{{\mathrm{{new}}}} = b - \eta\, \dfrac{{\partial \mathcal{{L}}}}{{\partial b}} = {b:.2f} - {eta:.2f}({dL_db:.2f}) = {b_new:.2f}$")
                line4.set_text(rf"New predictions:  $\hat{{y}}_{{\mathrm{{new}}}}=[{y_pred_new[0]:.2f},\ {y_pred_new[1]:.2f},\ {y_pred_new[2]:.2f}]$")
                line5.set_text(rf"Old loss:  ${mse:.2f}$   →   New loss:  ${mse_new:.2f}$")
                line6.set_text(r"The dashed line is the updated model after one learning step.")

            return (
                title_text, line1, line2, line3, line4, line5, line6,
                true_scatter, pred_scatter, pred_scatter_new, pred_line, pred_line_new,
                *error_lines, *error_labels
            )

        anim = FuncAnimation(fig, update, frames=len(stages), interval=interval, blit=False, repeat=repeat)
        plt.close(fig)
        return HTML(anim.to_jshtml())

    @classmethod
    def activation_functions_plot(cls, figsize=(15, 4.8), wspace=0.45):
        """
        Static plot of sigmoid, tanh, and ReLU with LaTeX formulas.
        Returns (fig, axes).
        """
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        def relu(z):
            return np.maximum(0, z)

        z_vals = np.linspace(-4, 4, 300)

        functions = [
            ("Sigmoid", sigmoid(z_vals), r"$a=\dfrac{1}{1+e^{-z}}$"),
            ("tanh", np.tanh(z_vals), r"$a=\tanh(z)$"),
            ("ReLU", relu(z_vals), r"$a=\max(0,z)$"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        fig.subplots_adjust(wspace=0.45)

        for i, (ax, (name, yvals, formula)) in enumerate(zip(axes, functions)):
            ax.plot(z_vals, yvals, linewidth=2)
            ax.axhline(0, linewidth=1, color="gray", linestyle="--")
            ax.axvline(0, linewidth=1, color="gray", linestyle="--")

            ax.set_title(name, fontsize=16)
            ax.set_xlabel(r"$z$", fontsize=14)

            ax.set_ylabel(r"$a=\varphi(z)$", fontsize=14)

            ax.text(
                0.05, 0.93, formula,
                transform=ax.transAxes,
                fontsize=13,
                va="top"
            )

            ax.tick_params(labelsize=12)

        return fig, axes

    @classmethod
    def loss_landscape_single_neuron(cls,
        x=None,
        y_true=None,
        w0=0.2,
        b0=2.5,
        eta=0.08,
        n_steps=50,
        w_range=(-0.5, 3.2),
        b_range=(-1.5, 3.5),
        grid_size=220,
        interval=900,
        repeat=True,
    ):
        """
        Animate gradient descent on the MSE loss landscape of a single linear neuron:
            y_hat = w x + b

        Returns
        -------
        IPython.display.HTML
            HTML animation for Jupyter notebooks.
        """
        if x is None:
            x = np.array([1.0, 2.0, 3.0])
        if y_true is None:
            y_true = np.array([2.0, 4.0, 6.0])

        x = np.asarray(x, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        N = len(x)

        def loss(w, b):
            y_hat = w * x + b
            return np.mean((y_hat - y_true) ** 2)

        def grad(w, b):
            err = w * x + b - y_true
            dL_dw = (2.0 / N) * np.sum(err * x)
            dL_db = (2.0 / N) * np.sum(err)
            return dL_dw, dL_db

        # -----------------------------
        # Precompute GD trajectory
        # -----------------------------
        ws = [w0]
        bs = [b0]
        losses = [loss(w0, b0)]
        grads = [grad(w0, b0)]

        w, b = w0, b0
        for _ in range(n_steps):
            dL_dw, dL_db = grad(w, b)
            w = w - eta * dL_dw
            b = b - eta * dL_db
            ws.append(w)
            bs.append(b)
            losses.append(loss(w, b))
            grads.append(grad(w, b))

        ws = np.array(ws)
        bs = np.array(bs)
        losses = np.array(losses)

        # -----------------------------
        # Loss grid
        # -----------------------------
        w_vals = np.linspace(*w_range, grid_size)
        b_vals = np.linspace(*b_range, grid_size)
        W, B = np.meshgrid(w_vals, b_vals)
        Z = np.zeros_like(W)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                Z[i, j] = loss(W[i, j], B[i, j])

        # -----------------------------
        # Figure layout
        # -----------------------------
        fig = plt.figure(figsize=(13, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.15], wspace=0.32)
        ax_text = fig.add_subplot(gs[0, 0])
        ax_plot = fig.add_subplot(gs[0, 1])
        ax_text.axis("off")
        ax_plot.set_box_aspect(1)
        fig.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.12)

        title_text = ax_text.text(
            0.02, 0.95, "",
            ha="left", va="top",
            fontsize=cls.TITLE_SIZE, weight="bold",
            transform=ax_text.transAxes
        )
        line1 = ax_text.text(0.02, 0.80, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line2 = ax_text.text(0.02, 0.68, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line3 = ax_text.text(0.02, 0.56, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line4 = ax_text.text(0.02, 0.42, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line5 = ax_text.text(0.02, 0.28, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line6 = ax_text.text(0.02, 0.14, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)

        # Landscape
        cf = ax_plot.contourf(W, B, Z, levels=35)
        ax_plot.contour(W, B, Z, levels=12, linewidths=0.8)
        cbar = fig.colorbar(cf, ax=ax_plot, fraction=0.046, pad=0.04, label=r"$\mathcal{L}(w,b)$")
        cbar.set_label(r"$\mathcal{L}(w,b)$", fontsize=cls.LABEL_SIZE)
        cbar.ax.tick_params(labelsize=cls.TICK_LABEL_SIZE)

        ax_plot.set_title("Loss landscape of a single linear neuron", fontsize=cls.TITLE_SIZE, pad=12)
        ax_plot.set_xlabel(r"$w$", fontsize=cls.LABEL_SIZE)
        ax_plot.set_ylabel(r"$b$", fontsize=cls.LABEL_SIZE)
        ax_plot.tick_params(labelsize=cls.TICK_LABEL_SIZE)

        # Artists
        path_line, = ax_plot.plot([], [], linewidth=2)
        current_point, = ax_plot.plot([], [], "o", markersize=9)
        next_point, = ax_plot.plot([], [], "o", markersize=6, alpha=0.8)
        grad_arrow = ax_plot.quiver([], [], [], [], angles="xy", scale_units="xy", scale=1, color="white")

        def update(frame):
            nonlocal grad_arrow

            t = frame
            w_t, b_t = ws[t], bs[t]
            L_t = losses[t]
            dL_dw, dL_db = grads[t]

            title_text.set_text(f"Gradient descent — step {t}")
            line1.set_text(r"Model:  $\hat{y}_i = w x_i + b$")
            line2.set_text(r"$\mathcal{L}(w,b)=\dfrac{1}{N}\sum_{i=1}^{N}(w x_i+b-y_i)^2$")
            line3.set_text(rf"Current parameters:  $w={w_t:.3f},\ b={b_t:.3f}$")
            line4.set_text(rf"Current loss:  $\mathcal{{L}}={L_t:.4f}$")
            line5.set_text(rf"Gradient:  $\nabla \mathcal{{L}} = ({dL_dw:.3f},\ {dL_db:.3f})$")
            if t < len(ws) - 1:
                line6.set_text(
                    rf"Update:  $(w,b)\leftarrow(w,b)-\eta\nabla\mathcal{{L}}$, "
                    rf"with $\eta={eta:.3f}$"
                )
            else:
                line6.set_text(r"Trajectory approaches the minimum of the loss surface.")

            path_line.set_data(ws[:t+1], bs[:t+1])
            current_point.set_data([w_t], [b_t])

            if t < len(ws) - 1:
                next_point.set_data([ws[t+1]], [bs[t+1]])
            else:
                next_point.set_data([], [])

            # Replace quiver each frame for simplicity
            grad_arrow.remove()
            grad_arrow = ax_plot.quiver(
                w_t, b_t, dL_dw, dL_db,
                angles="xy", scale_units="xy", scale=18, width=0.006, color="white"
            )

            return (
                title_text, line1, line2, line3, line4, line5, line6,
                path_line, current_point, next_point, grad_arrow
            )

        anim = FuncAnimation(
            fig,
            update,
            frames=len(ws),
            interval=interval,
            blit=False,
            repeat=repeat
        )

        plt.close(fig)
        return HTML(anim.to_jshtml())

    @classmethod
    def loss_landscape_learning_rates(cls,
        x=None,
        y_true=None,
        w0=0.2,
        b0=2.5,
        etas=(0.03, 0.08, 0.15),
        n_steps=50,
        w_range=(-0.5, 3.2),
        b_range=(-1.5, 3.5),
        grid_size=220,
        interval=900,
        repeat=True,
    ):
        """
        Compare several learning rates on the MSE loss landscape
        of a single linear neuron.

        Returns
        -------
        IPython.display.HTML
            HTML animation for Jupyter notebooks.
        """
        if x is None:
            x = np.array([1.0, 2.0, 3.0])
        if y_true is None:
            y_true = np.array([2.0, 4.0, 6.0])

        x = np.asarray(x, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        N = len(x)

        def loss(w, b):
            y_hat = w * x + b
            return np.mean((y_hat - y_true) ** 2)

        def grad(w, b):
            err = w * x + b - y_true
            dL_dw = (2.0 / N) * np.sum(err * x)
            dL_db = (2.0 / N) * np.sum(err)
            return dL_dw, dL_db

        # -----------------------------
        # Precompute trajectories
        # -----------------------------
        trajectories = []
        for eta in etas:
            ws = [w0]
            bs = [b0]
            losses = [loss(w0, b0)]

            w, b = w0, b0
            for _ in range(n_steps):
                dL_dw, dL_db = grad(w, b)
                w = w - eta * dL_dw
                b = b - eta * dL_db
                ws.append(w)
                bs.append(b)
                losses.append(loss(w, b))

            trajectories.append({
                "eta": eta,
                "ws": np.array(ws),
                "bs": np.array(bs),
                "losses": np.array(losses),
            })

        # -----------------------------
        # Loss grid
        # -----------------------------
        w_vals = np.linspace(*w_range, grid_size)
        b_vals = np.linspace(*b_range, grid_size)
        W, B = np.meshgrid(w_vals, b_vals)
        Z = np.zeros_like(W)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                Z[i, j] = loss(W[i, j], B[i, j])

        # -----------------------------
        # Figure layout
        # -----------------------------
        fig = plt.figure(figsize=(13, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.15], wspace=0.32)
        ax_text = fig.add_subplot(gs[0, 0])
        ax_plot = fig.add_subplot(gs[0, 1])
        ax_text.axis("off")
        ax_plot.set_box_aspect(1)
        fig.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.12)

        title_text = ax_text.text(
            0.02, 0.95, "",
            ha="left", va="top",
            fontsize=cls.TITLE_SIZE, weight="bold",
            transform=ax_text.transAxes
        )
        line1 = ax_text.text(0.02, 0.80, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line2 = ax_text.text(0.02, 0.68, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line3 = ax_text.text(0.02, 0.56, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line4 = ax_text.text(0.02, 0.42, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line5 = ax_text.text(0.02, 0.28, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line6 = ax_text.text(0.02, 0.14, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)

        cf = ax_plot.contourf(W, B, Z, levels=35)
        ax_plot.contour(W, B, Z, levels=12, linewidths=0.8)
        cbar = fig.colorbar(cf, ax=ax_plot, fraction=0.046, pad=0.04, label=r"$\mathcal{L}(w,b)$")
        cbar.set_label(r"$\mathcal{L}(w,b)$", fontsize=cls.LABEL_SIZE)
        cbar.ax.tick_params(labelsize=cls.TICK_LABEL_SIZE)

        ax_plot.set_title("Learning-rate comparison on the loss landscape", fontsize=cls.TITLE_SIZE, pad=12)
        ax_plot.set_xlabel(r"$w$", fontsize=cls.LABEL_SIZE)
        ax_plot.set_ylabel(r"$b$", fontsize=cls.LABEL_SIZE)
        ax_plot.tick_params(labelsize=cls.TICK_LABEL_SIZE)

        labels = [rf"$\eta={eta:.2f}$" for eta in etas]
        path_lines = [ax_plot.plot([], [], linewidth=2, label=lab)[0] for lab in labels]
        current_points = [ax_plot.plot([], [], "o", markersize=8)[0] for _ in etas]
        ax_plot.legend(loc="upper right", fontsize=10, frameon=True)

        def update(frame):
            t = frame
            title_text.set_text(f"Learning-rate comparison — step {t}")
            line1.set_text(r"Same model and same initial point, different learning rates.")
            line2.set_text(r"$\hat{y}_i = w x_i + b$")
            line3.set_text(r"$\mathcal{L}(w,b)=\dfrac{1}{N}\sum_{i=1}^{N}(w x_i+b-y_i)^2$")
            line4.set_text(rf"Initial point:  $w_0={w0:.2f},\ b_0={b0:.2f}$")

            parts = []
            for tr in trajectories:
                idx = min(t, len(tr["losses"]) - 1)
                parts.append(rf"$\eta={tr['eta']:.2f}\,:\,\mathcal{{L}}={tr['losses'][idx]:.4f}$")
            line5.set_text("   ".join(parts[:2]))
            line6.set_text("   ".join(parts[2:]))

            for line, point, tr in zip(path_lines, current_points, trajectories):
                idx = min(t, len(tr["ws"]) - 1)
                line.set_data(tr["ws"][:idx+1], tr["bs"][:idx+1])
                point.set_data([tr["ws"][idx]], [tr["bs"][idx]])

            return (
                title_text, line1, line2, line3, line4, line5, line6,
                *path_lines, *current_points
            )

        anim = FuncAnimation(
            fig,
            update,
            frames=n_steps + 1,
            interval=interval,
            blit=False,
            repeat=repeat
        )

        plt.close(fig)
        return HTML(anim.to_jshtml())

    @classmethod
    def backpropagation_example(cls, interval=2400, repeat=True):
        # =========================================================
        # Backpropagation example: full forward, backward,
        # and one parameter-update step
        # Tiny 1-2-1 network, one sample
        # =========================================================

        # -----------------------------
        # Data: one input-target pair
        # -----------------------------
        x = 1.5
        y = 0.8
        eta = 0.025

        # -----------------------------
        # Network parameters
        # 1 input -> 2 hidden -> 1 output
        # -----------------------------
        W1 = np.array([[0.5],
                    [-0.4]])          # shape (2,1)

        b1 = np.array([[0.1],
                    [0.2]])           # shape (2,1)

        W2 = np.array([[0.7, -1.1]])     # shape (1,2)
        b2 = np.array([[0.05]])          # shape (1,1)

        # -----------------------------
        # Helper functions
        # -----------------------------
        def tanh(z):
            return np.tanh(z)

        def tanh_prime(z):
            return 1.0 - np.tanh(z) ** 2

        def forward_pass(x_scalar, y_scalar, W1_, b1_, W2_, b2_):
            x_col_ = np.array([[x_scalar]])
            y_col_ = np.array([[y_scalar]])

            z1_ = W1_ @ x_col_ + b1_
            a1_ = tanh(z1_)
            z2_ = W2_ @ a1_ + b2_
            y_hat_ = z2_.copy()          # linear output
            L_ = (y_hat_ - y_col_) ** 2

            return {
                "x_col": x_col_,
                "y_col": y_col_,
                "z1": z1_,
                "a1": a1_,
                "z2": z2_,
                "y_hat": y_hat_,
                "L": L_,
            }

        # -----------------------------
        # Initial forward pass
        # -----------------------------
        cache = forward_pass(x, y, W1, b1, W2, b2)

        x_col = cache["x_col"]
        y_col = cache["y_col"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        z2 = cache["z2"]
        y_hat = cache["y_hat"]
        L = cache["L"]

        # -----------------------------
        # Backward pass
        # -----------------------------
        dL_dyhat = 2.0 * (y_hat - y_col)      # shape (1,1)
        dL_dz2 = dL_dyhat * 1.0               # linear output

        dL_dW2 = dL_dz2 @ a1.T                # shape (1,2)
        dL_db2 = dL_dz2                       # shape (1,1)

        dL_da1 = W2.T @ dL_dz2                # shape (2,1)
        dL_dz1 = dL_da1 * tanh_prime(z1)      # shape (2,1)

        dL_dW1 = dL_dz1 @ x_col.T             # shape (2,1)
        dL_db1 = dL_dz1                       # shape (2,1)

        # -----------------------------
        # One gradient descent update
        # -----------------------------
        W1_new = W1 - eta * dL_dW1
        b1_new = b1 - eta * dL_db1
        W2_new = W2 - eta * dL_dW2
        b2_new = b2 - eta * dL_db2

        # -----------------------------
        # New forward pass after update
        # -----------------------------
        cache_new = forward_pass(x, y, W1_new, b1_new, W2_new, b2_new)

        z1_new = cache_new["z1"]
        a1_new = cache_new["a1"]
        z2_new = cache_new["z2"]
        y_hat_new = cache_new["y_hat"]
        L_new = cache_new["L"]

        # -----------------------------
        # Animation stages
        # -----------------------------
        stages = [
            "Show input and target",
            r"Compute hidden pre-activations $z^{(1)}$",
            r"Apply hidden activation $a^{(1)} = \tanh(z^{(1)})$",
            r"Compute output $z^{(2)}$ and prediction $\hat{y}$",
            r"Compute loss $\mathcal{L} = (\hat{y} - y)^2$",
            r"Start backward pass: $\frac{\partial \mathcal{L}}{\partial \hat{y}}$ and $\frac{\partial \mathcal{L}}{\partial z^{(2)}}$",
            r"Gradients of output layer: $\frac{\partial \mathcal{L}}{\partial W^{(2)}}$, $\frac{\partial \mathcal{L}}{\partial b^{(2)}}$",
            r"Backpropagate to hidden activation: $\frac{\partial \mathcal{L}}{\partial a^{(1)}}$",
            r"Backpropagate through tanh: $\frac{\partial \mathcal{L}}{\partial z^{(1)}}$",
            r"Gradients of first layer: $\frac{\partial \mathcal{L}}{\partial W^{(1)}}$, $\frac{\partial \mathcal{L}}{\partial b^{(1)}}$",
            "Update the parameters",
            "Run a new forward pass with updated parameters",
        ]

        fig = plt.figure(figsize=(13, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.1], wspace=0.32)

        ax_text = fig.add_subplot(gs[0, 0])
        ax_net = fig.add_subplot(gs[0, 1])

        ax_text.axis("off")
        ax_net.axis("off")

        fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.08)

        title_text = ax_text.text(
            0.02, 0.95, "", ha="left", va="top",
            fontsize=cls.TITLE_SIZE, weight="bold", transform=ax_text.transAxes
        )

        line1 = ax_text.text(0.02, 0.82, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line2 = ax_text.text(0.02, 0.70, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line3 = ax_text.text(0.02, 0.58, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line4 = ax_text.text(0.02, 0.46, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line5 = ax_text.text(0.02, 0.34, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line6 = ax_text.text(0.02, 0.20, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)

        # -----------------------------
        # Draw network diagram
        # -----------------------------
        pos = {
            "x":    (0.10, 0.50),
            "z1_1": (0.42, 0.72),
            "z1_2": (0.42, 0.28),
            "a1_1": (0.63, 0.72),
            "a1_2": (0.63, 0.28),
            "yhat": (0.86, 0.50),
        }

        edges = [
            ("x", "z1_1"),
            ("x", "z1_2"),
            ("z1_1", "a1_1"),
            ("z1_2", "a1_2"),
            ("a1_1", "yhat"),
            ("a1_2", "yhat"),
        ]

        node_artists = {}
        edge_artists = []

        for start, end in edges:
            x0, y0 = pos[start]
            x1_, y1_ = pos[end]
            line, = ax_net.plot([x0, x1_], [y0, y1_], linewidth=2, alpha=0.35)
            edge_artists.append(((start, end), line))

        labels = {
            "x": r"$x$",
            "z1_1": r"$z^{(1)}_1$",
            "z1_2": r"$z^{(1)}_2$",
            "a1_1": r"$a^{(1)}_1$",
            "a1_2": r"$a^{(1)}_2$",
            "yhat": r"$\hat{y}$",
        }

        for name, (xp, yp) in pos.items():
            pt, = ax_net.plot(
                [xp], [yp],
                "o",
                markersize=50,
                markerfacecolor="#ece6e6",
                markeredgecolor="black",
                markeredgewidth=1.5
            )
            txt = ax_net.text(xp, yp, labels[name], ha="center", va="center", fontsize=18)
            node_artists[name] = (pt, txt)

        ax_net.set_xlim(0.0, 1.0)
        ax_net.set_ylim(0.0, 1.0)

        # -----------------------------
        # Highlight helpers
        # -----------------------------
        def reset_network_style():
            for _, line in edge_artists:
                line.set_alpha(0.25)
                line.set_linewidth(2)

            for pt, _ in node_artists.values():
                pt.set_markersize(50)
                pt.set_alpha(1.0)

        def highlight_nodes(names, size=22):
            for name in names:
                pt, _ = node_artists[name]
                pt.set_markersize(50)
                pt.set_alpha(1.0)

        def highlight_edges(pairs, lw=3.5):
            for (start, end), line in edge_artists:
                if (start, end) in pairs:
                    line.set_alpha(1.0)
                    line.set_linewidth(lw)

        # -----------------------------
        # Animation update
        # -----------------------------
        def update(frame):
            reset_network_style()

            title_text.set_text(stages[frame])
            line1.set_text("")
            line2.set_text("")
            line3.set_text("")
            line4.set_text("")
            line5.set_text("")
            line6.set_text("")

            if frame == 0:
                highlight_nodes(["x"])
                line1.set_text(rf"Input:  $x = {x:.2f}$")
                line2.set_text(rf"Target:  $y = {y:.2f}$")
                line3.set_text(rf"$W^{{(1)}} = \left[{W1[0,0]:.2f},\ {W1[1,0]:.2f}\right]^T$")
                line4.set_text(rf"$b^{{(1)}} = \left[{b1[0,0]:.2f},\ {b1[1,0]:.2f}\right]^T$")
                line5.set_text(rf"$W^{{(2)}} = \left[{W2[0,0]:.2f},\ {W2[0,1]:.2f}\right], \quad b^{{(2)}} = {b2[0,0]:.2f}$")
                line6.set_text(r"The network has 2 hidden neurons and 1 output neuron.")

            elif frame == 1:
                highlight_nodes(["x", "z1_1", "z1_2"])
                highlight_edges([("x", "z1_1"), ("x", "z1_2")])
                line1.set_text(r"$z^{(1)} = W^{(1)}x + b^{(1)}$")
                line2.set_text(rf"$z_1^{{(1)}} = 0.5({x:.2f}) + 0.1 = {z1[0,0]:.4f}$")
                line3.set_text(rf"$z_2^{{(1)}} = -0.4({x:.2f}) + 0.2 = {z1[1,0]:.4f}$")
                line4.set_text(rf"$z^{{(1)}} = \left[{z1[0,0]:.4f},\ {z1[1,0]:.4f}\right]^T$")

            elif frame == 2:
                highlight_nodes(["z1_1", "z1_2", "a1_1", "a1_2"])
                highlight_edges([("z1_1", "a1_1"), ("z1_2", "a1_2")])
                line1.set_text(r"$a^{(1)} = \tanh(z^{(1)})$")
                line2.set_text(rf"$a_1^{{(1)}} = \tanh({z1[0,0]:.4f}) = {a1[0,0]:.4f}$")
                line3.set_text(rf"$a_2^{{(1)}} = \tanh({z1[1,0]:.4f}) = {a1[1,0]:.4f}$")
                line4.set_text(rf"$a^{{(1)}} = \left[{a1[0,0]:.4f},\ {a1[1,0]:.4f}\right]^T$")

            elif frame == 3:
                highlight_nodes(["a1_1", "a1_2", "yhat"])
                highlight_edges([("a1_1", "yhat"), ("a1_2", "yhat")])
                line1.set_text(r"$z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}$")
                line2.set_text(
                    rf"$z^{{(2)}} = 0.7({a1[0,0]:.4f}) + (-1.1)({a1[1,0]:.4f}) + 0.05$"
                )
                line3.set_text(rf"$z^{{(2)}} = {z2[0,0]:.4f}$")
                line4.set_text(rf"$\hat{{y}} = z^{{(2)}} = {y_hat[0,0]:.4f}$")

            elif frame == 4:
                highlight_nodes(["yhat"])
                line1.set_text(r"$\mathcal{L} = (\hat{y} - y)^2$")
                line2.set_text(rf"$\mathcal{{L}} = ({y_hat[0,0]:.4f} - {y:.4f})^2$")
                line3.set_text(rf"$\mathcal{{L}} = {L[0,0]:.6f}$")
                line4.set_text(r"This completes the forward pass.")
                line5.set_text(r"Now we move backward through the computation graph.")

            elif frame == 5:
                highlight_nodes(["yhat"])
                line1.set_text(r"Start backward pass from the loss.")
                line2.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial \hat{y}} = 2(\hat{y} - y)$")
                line3.set_text(rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial \hat{{y}}}} = {dL_dyhat[0,0]:.4f}$")
                line4.set_text(r"Because the output is linear,")
                line5.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial z^{(2)}} = \dfrac{\partial \mathcal{L}}{\partial \hat{y}}$")
                line6.set_text(rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial z^{{(2)}}}} = {dL_dz2[0,0]:.4f}$")

            elif frame == 6:
                highlight_nodes(["a1_1", "a1_2", "yhat"])
                highlight_edges([("a1_1", "yhat"), ("a1_2", "yhat")])
                line1.set_text(r"Output-layer gradients:")
                line2.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial W^{(2)}} = \dfrac{\partial \mathcal{L}}{\partial z^{(2)}} a^{(1)\top}$")
                line3.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial W^{{(2)}}}} = "
                    rf"\left[{dL_dW2[0,0]:.4f},\ {dL_dW2[0,1]:.4f}\right]$"
                )
                line4.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial b^{(2)}} = \dfrac{\partial \mathcal{L}}{\partial z^{(2)}}$")
                line5.set_text(rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial b^{{(2)}}}} = {dL_db2[0,0]:.4f}$")

            elif frame == 7:
                highlight_nodes(["a1_1", "a1_2"])
                highlight_edges([("a1_1", "yhat"), ("a1_2", "yhat")])
                line1.set_text(r"Backpropagate to the hidden activation:")
                line2.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial a^{(1)}} = W^{(2)\top}\dfrac{\partial \mathcal{L}}{\partial z^{(2)}}$")
                line3.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial a^{{(1)}}}} = "
                    rf"\left[{dL_da1[0,0]:.4f},\ {dL_da1[1,0]:.4f}\right]^T$"
                )
                line4.set_text(r"The error signal is now distributed to both hidden neurons.")

            elif frame == 8:
                highlight_nodes(["z1_1", "z1_2", "a1_1", "a1_2"])
                highlight_edges([("z1_1", "a1_1"), ("z1_2", "a1_2")])
                line1.set_text(r"Backpropagate through the hidden nonlinearity:")
                line2.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial z^{(1)}} = \dfrac{\partial \mathcal{L}}{\partial a^{(1)}} \odot (1-\tanh^2(z^{(1)}))$")
                line3.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial z^{{(1)}}}} = "
                    rf"\left[{dL_dz1[0,0]:.4f},\ {dL_dz1[1,0]:.4f}\right]$"
                )
                line4.set_text(r"Here the chain rule multiplies by the local derivative of tanh.")

            elif frame == 9:
                highlight_nodes(["x", "z1_1", "z1_2"])
                highlight_edges([("x", "z1_1"), ("x", "z1_2")])
                line1.set_text(r"First-layer gradients:")
                line2.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial W^{(1)}} = \dfrac{\partial \mathcal{L}}{\partial z^{(1)}} x^\top$")
                line3.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial W^{{(1)}}}} = "
                    rf"\left[{dL_dW1[0,0]:.4f},\ {dL_dW1[1,0]:.4f}\right]^T$"
                )
                line4.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial b^{{(1)}}}} = "
                    rf"\left[{dL_db1[0,0]:.4f},\ {dL_db1[1,0]:.4f}\right]^T$"
                )
                line5.set_text(r"This completes backpropagation.")
                line6.set_text(r"We now have gradients for every trainable parameter in the network.")

            elif frame == 10:
                highlight_nodes(["x", "z1_1", "z1_2", "a1_1", "a1_2", "yhat"])
                highlight_edges(edges)

                line1.set_text(r"Gradient descent update:")
                line2.set_text(r"$W^{(1)}_{new} = W^{(1)} - \eta\, \dfrac{\partial \mathcal{L}}{\partial W^{(1)}};\qquad b^{(1)}_{new} = b^{(1)} - \eta\, \dfrac{\partial \mathcal{L}}{\partial b^{(1)}}$")
                line3.set_text(
                    rf"$W^{{(1)}}_{{new}} = "
                    rf"\left[{W1_new[0,0]:.4f},\ {W1_new[1,0]:.4f}\right]$"
                )
                line4.set_text(
                    rf"$b^{{(1)}}_{{new}} = "
                    rf"\left[{b1_new[0,0]:.4f},\ {b1_new[1,0]:.4f}\right]$"
                )
                line5.set_text(
                    rf"$W^{{(2)}}_{{new}} = "
                    rf"\left[{W2_new[0,0]:.4f},\ {W2_new[0,1]:.4f}\right], \quad "
                    rf"b^{{(2)}}_{{new}} = {b2_new[0,0]:.4f}$"
                )
                line6.set_text(rf"Learning rate: $\eta = {eta:.2f}$")

            elif frame == 11:
                highlight_nodes(["x", "z1_1", "z1_2", "a1_1", "a1_2", "yhat"])
                highlight_edges(edges)

                line1.set_text(r"New forward pass with updated parameters:")
                line2.set_text(
                    rf"$z^{{(1)}}_{{new}} = "
                    rf"\left[{z1_new[0,0]:.4f},\ {z1_new[1,0]:.4f}\right]^T$"
                )
                line3.set_text(
                    rf"$a^{{(1)}}_{{new}} = "
                    rf"\left[{a1_new[0,0]:.4f},\ {a1_new[1,0]:.4f}\right]^T$"
                )
                line4.set_text(
                    rf"Old prediction: $\hat{{y}} = {y_hat[0,0]:.4f}$"
                    rf" $\rightarrow$ New prediction: $\hat{{y}}_{{new}} = {y_hat_new[0,0]:.4f}$"
                )
                line5.set_text(
                    rf"Old loss: $\mathcal{{L}} = {L[0,0]:.6f}$"
                    rf" $\rightarrow$ New loss: $\mathcal{{L}}_{{new}} = {L_new[0,0]:.6f}$"
                )
                line6.set_text(r"The loss decreases because we moved opposite to the gradient.")

            return (
                title_text, line1, line2, line3, line4, line5, line6,
                *[line for _, line in edge_artists],
                *[pt for pt, _ in node_artists.values()]
            )

        anim = FuncAnimation(
            fig,
            update,
            frames=len(stages),
            interval=interval,
            blit=False,
            repeat=repeat
        )

        plt.close(fig)
        return HTML(anim.to_jshtml())

    @classmethod
    def backpropagation_example_1221(cls, interval=2400, repeat=True):
        # =========================================================
        # Backpropagation example: full forward, backward,
        # and one parameter-update step
        # Tiny 1-2-2-1 network, one sample
        # =========================================================

        # -----------------------------
        # Data: one input-target pair
        # -----------------------------
        x = 1.2
        y = 0.5
        eta = 0.02

        # -----------------------------
        # Network parameters
        # 1 input -> 2 hidden -> 2 hidden -> 1 output
        # -----------------------------
        W1 = np.array([[0.6],
                    [-0.5]])                # shape (2,1)

        b1 = np.array([[0.1],
                    [-0.2]])                # shape (2,1)

        W2 = np.array([[0.4, -0.7],
                    [0.9,  0.3]])           # shape (2,2)

        b2 = np.array([[0.05],
                    [-0.10]])               # shape (2,1)

        W3 = np.array([[1.1, -0.6]])           # shape (1,2)
        b3 = np.array([[0.02]])                # shape (1,1)

        # -----------------------------
        # Helper functions
        # -----------------------------
        def tanh(z):
            return np.tanh(z)

        def tanh_prime(z):
            return 1.0 - np.tanh(z) ** 2

        def forward_pass(x_scalar, y_scalar, W1_, b1_, W2_, b2_, W3_, b3_):
            x_col_ = np.array([[x_scalar]])
            y_col_ = np.array([[y_scalar]])

            z1_ = W1_ @ x_col_ + b1_
            a1_ = tanh(z1_)

            z2_ = W2_ @ a1_ + b2_
            a2_ = tanh(z2_)

            z3_ = W3_ @ a2_ + b3_
            y_hat_ = z3_.copy()          # linear output
            L_ = (y_hat_ - y_col_) ** 2

            return {
                "x_col": x_col_,
                "y_col": y_col_,
                "z1": z1_,
                "a1": a1_,
                "z2": z2_,
                "a2": a2_,
                "z3": z3_,
                "y_hat": y_hat_,
                "L": L_,
            }

        # -----------------------------
        # Initial forward pass
        # -----------------------------
        cache = forward_pass(x, y, W1, b1, W2, b2, W3, b3)

        x_col = cache["x_col"]
        y_col = cache["y_col"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        z2 = cache["z2"]
        a2 = cache["a2"]
        z3 = cache["z3"]
        y_hat = cache["y_hat"]
        L = cache["L"]

        # -----------------------------
        # Backward pass
        # -----------------------------
        dL_dyhat = 2.0 * (y_hat - y_col)      # shape (1,1)
        dL_dz3 = dL_dyhat * 1.0               # linear output

        dL_dW3 = dL_dz3 @ a2.T                # shape (1,2)
        dL_db3 = dL_dz3                       # shape (1,1)

        dL_da2 = W3.T @ dL_dz3                # shape (2,1)
        dL_dz2 = dL_da2 * tanh_prime(z2)      # shape (2,1)

        dL_dW2 = dL_dz2 @ a1.T                # shape (2,2)
        dL_db2 = dL_dz2                       # shape (2,1)

        dL_da1 = W2.T @ dL_dz2                # shape (2,1)
        dL_dz1 = dL_da1 * tanh_prime(z1)      # shape (2,1)

        dL_dW1 = dL_dz1 @ x_col.T             # shape (2,1)
        dL_db1 = dL_dz1                       # shape (2,1)

        # -----------------------------
        # One gradient descent update
        # -----------------------------
        W1_new = W1 - eta * dL_dW1
        b1_new = b1 - eta * dL_db1

        W2_new = W2 - eta * dL_dW2
        b2_new = b2 - eta * dL_db2

        W3_new = W3 - eta * dL_dW3
        b3_new = b3 - eta * dL_db3

        # -----------------------------
        # New forward pass after update
        # -----------------------------
        cache_new = forward_pass(x, y, W1_new, b1_new, W2_new, b2_new, W3_new, b3_new)

        z1_new = cache_new["z1"]
        a1_new = cache_new["a1"]
        z2_new = cache_new["z2"]
        a2_new = cache_new["a2"]
        z3_new = cache_new["z3"]
        y_hat_new = cache_new["y_hat"]
        L_new = cache_new["L"]

        # -----------------------------
        # Animation stages
        # -----------------------------
        stages = [
            "Show input and target",
            r"Compute first hidden pre-activations $z^{(1)}$",
            r"Apply first hidden activation $a^{(1)} = \tanh(z^{(1)})$",
            r"Compute second hidden pre-activations $z^{(2)}$",
            r"Apply second hidden activation $a^{(2)} = \tanh(z^{(2)})$",
            r"Compute output $z^{(3)}$ and prediction $\hat{y}$",
            r"Compute loss $\mathcal{L} = (\hat{y} - y)^2$",
            r"Start backward pass: $\frac{\partial \mathcal{L}}{\partial \hat{y}}$ and $\frac{\partial \mathcal{L}}{\partial z^{(3)}}$",
            r"Gradients of output layer: $\frac{\partial \mathcal{L}}{\partial W^{(3)}}$, $\frac{\partial \mathcal{L}}{\partial b^{(3)}}$",
            r"Backpropagate to second hidden layer: $\frac{\partial \mathcal{L}}{\partial a^{(2)}}$, $\frac{\partial \mathcal{L}}{\partial z^{(2)}}$",
            r"Gradients of second layer: $\frac{\partial \mathcal{L}}{\partial W^{(2)}}$, $\frac{\partial \mathcal{L}}{\partial b^{(2)}}$",
            r"Backpropagate to first hidden layer: $\frac{\partial \mathcal{L}}{\partial a^{(1)}}$, $\frac{\partial \mathcal{L}}{\partial z^{(1)}}$",
            r"Gradients of first layer: $\frac{\partial \mathcal{L}}{\partial W^{(1)}}$, $\frac{\partial \mathcal{L}}{\partial b^{(1)}}$",
            "Update the parameters",
            "Run a new forward pass with updated parameters",
        ]

        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.15], wspace=0.32)

        ax_text = fig.add_subplot(gs[0, 0])
        ax_net = fig.add_subplot(gs[0, 1])

        ax_text.axis("off")
        ax_net.axis("off")

        fig.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.08)

        title_text = ax_text.text(
            0.02, 0.95, "", ha="left", va="top",
            fontsize=cls.TITLE_SIZE, weight="bold", transform=ax_text.transAxes
        )

        line1 = ax_text.text(0.02, 0.84, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line2 = ax_text.text(0.02, 0.73, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line3 = ax_text.text(0.02, 0.62, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line4 = ax_text.text(0.02, 0.51, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line5 = ax_text.text(0.02, 0.40, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)
        line6 = ax_text.text(0.02, 0.27, "", fontsize=cls.TEXT_SIZE, transform=ax_text.transAxes)

        # -----------------------------
        # Draw network diagram
        # -----------------------------
        pos = {
            "x":    (0.08, 0.50),

            "z1_1": (0.28, 0.72),
            "z1_2": (0.28, 0.28),

            "a1_1": (0.42, 0.72),
            "a1_2": (0.42, 0.28),

            "z2_1": (0.58, 0.72),
            "z2_2": (0.58, 0.28),

            "a2_1": (0.72, 0.72),
            "a2_2": (0.72, 0.28),

            "yhat": (0.90, 0.50),
        }

        edges = [
            ("x", "z1_1"),
            ("x", "z1_2"),

            ("z1_1", "a1_1"),
            ("z1_2", "a1_2"),

            ("a1_1", "z2_1"),
            ("a1_1", "z2_2"),
            ("a1_2", "z2_1"),
            ("a1_2", "z2_2"),

            ("z2_1", "a2_1"),
            ("z2_2", "a2_2"),

            ("a2_1", "yhat"),
            ("a2_2", "yhat"),
        ]

        node_artists = {}
        edge_artists = []

        for start, end in edges:
            x0, y0 = pos[start]
            x1_, y1_ = pos[end]
            line, = ax_net.plot([x0, x1_], [y0, y1_], linewidth=2, alpha=0.35)
            edge_artists.append(((start, end), line))

        labels = {
            "x": r"$x$",

            "z1_1": r"$z^{(1)}_1$",
            "z1_2": r"$z^{(1)}_2$",

            "a1_1": r"$a^{(1)}_1$",
            "a1_2": r"$a^{(1)}_2$",

            "z2_1": r"$z^{(2)}_1$",
            "z2_2": r"$z^{(2)}_2$",

            "a2_1": r"$a^{(2)}_1$",
            "a2_2": r"$a^{(2)}_2$",

            "yhat": r"$\hat{y}$",
        }

        for name, (xp, yp) in pos.items():
            pt, = ax_net.plot(
                [xp], [yp],
                "o",
                markersize=40,
                markerfacecolor="#ece6e6",
                markeredgecolor="black",
                markeredgewidth=1.5
            )
            txt = ax_net.text(xp, yp, labels[name], ha="center", va="center", fontsize=15)
            node_artists[name] = (pt, txt)

        ax_net.set_xlim(0.0, 1.0)
        ax_net.set_ylim(0.0, 1.0)

        # -----------------------------
        # Highlight helpers
        # -----------------------------
        def reset_network_style():
            for _, line in edge_artists:
                line.set_alpha(0.25)
                line.set_linewidth(2)

            for pt, _ in node_artists.values():
                pt.set_markersize(40)
                pt.set_alpha(1.0)

        def highlight_nodes(names):
            for name in names:
                pt, _ = node_artists[name]
                pt.set_markersize(40)
                pt.set_alpha(1.0)

        def highlight_edges(pairs, lw=3.5):
            for (start, end), line in edge_artists:
                if (start, end) in pairs:
                    line.set_alpha(1.0)
                    line.set_linewidth(lw)

        # -----------------------------
        # Animation update
        # -----------------------------
        def update(frame):
            reset_network_style()

            title_text.set_text(stages[frame])
            line1.set_text("")
            line2.set_text("")
            line3.set_text("")
            line4.set_text("")
            line5.set_text("")
            line6.set_text("")

            if frame == 0:
                highlight_nodes(["x"])
                line1.set_text(rf"Input:  $x = {x:.2f}$")
                line2.set_text(rf"Target:  $y = {y:.2f}$")
                line3.set_text(rf"$W^{{(1)}} = \left[{W1[0,0]:.2f},\ {W1[1,0]:.2f}\right]^T,\quad b^{{(1)}} = \left[{b1[0,0]:.2f},\ {b1[1,0]:.2f}\right]^T$")
                line4.set_text(rf"$W^{{(2)}} = \left[\left[{W2[0,0]:.2f},\, {W2[0,1]:.2f}\right],\,\left[{W2[1,0]:.2f},\, {W2[1,1]:.2f}\right]\right],\quad b^{{(2)}} = \left[{b2[0,0]:.2f},\ {b2[1,0]:.2f}\right]^T$")
                line5.set_text(rf"$W^{{(3)}} = \left[{W3[0,0]:.2f},\ {W3[0,1]:.2f}\right],\quad b^{{(3)}} = {b3[0,0]:.2f}$")
                line6.set_text(r"The network has two hidden layers with two neurons each.")

            elif frame == 1:
                highlight_nodes(["x", "z1_1", "z1_2"])
                highlight_edges([("x", "z1_1"), ("x", "z1_2")])
                line1.set_text(r"$z^{(1)} = W^{(1)}x + b^{(1)}$")
                line2.set_text(rf"$z_1^{{(1)}} = 0.6({x:.2f}) + 0.1 = {z1[0,0]:.4f}$")
                line3.set_text(rf"$z_2^{{(1)}} = -0.5({x:.2f}) - 0.2 = {z1[1,0]:.4f}$")
                line4.set_text(rf"$z^{{(1)}} = \left[{z1[0,0]:.4f},\ {z1[1,0]:.4f}\right]^T$")

            elif frame == 2:
                highlight_nodes(["z1_1", "z1_2", "a1_1", "a1_2"])
                highlight_edges([("z1_1", "a1_1"), ("z1_2", "a1_2")])
                line1.set_text(r"$a^{(1)} = \tanh(z^{(1)})$")
                line2.set_text(rf"$a_1^{{(1)}} = \tanh({z1[0,0]:.4f}) = {a1[0,0]:.4f}$")
                line3.set_text(rf"$a_2^{{(1)}} = \tanh({z1[1,0]:.4f}) = {a1[1,0]:.4f}$")
                line4.set_text(rf"$a^{{(1)}} = \left[{a1[0,0]:.4f},\ {a1[1,0]:.4f}\right]^T$")

            elif frame == 3:
                highlight_nodes(["a1_1", "a1_2", "z2_1", "z2_2"])
                highlight_edges([
                    ("a1_1", "z2_1"), ("a1_1", "z2_2"),
                    ("a1_2", "z2_1"), ("a1_2", "z2_2")
                ])
                line1.set_text(r"$z^{(2)} = W^{(2)}a^{(1)} + b^{(2)}$")
                line2.set_text(
                    rf"$z_1^{{(2)}} = 0.4({a1[0,0]:.4f}) + (-0.7)({a1[1,0]:.4f}) + 0.05 = {z2[0,0]:.4f}$"
                )
                line3.set_text(
                    rf"$z_2^{{(2)}} = 0.9({a1[0,0]:.4f}) + 0.3({a1[1,0]:.4f}) - 0.10 = {z2[1,0]:.4f}$"
                )
                line4.set_text(r"This layer mixes both activations from the previous layer.")
                line5.set_text(rf"$z^{{(2)}} = \left[{z2[0,0]:.4f},\ {z2[1,0]:.4f}\right]^T$")

            elif frame == 4:
                highlight_nodes(["z2_1", "z2_2", "a2_1", "a2_2"])
                highlight_edges([("z2_1", "a2_1"), ("z2_2", "a2_2")])
                line1.set_text(r"$a^{(2)} = \tanh(z^{(2)})$")
                line2.set_text(rf"$a_1^{{(2)}} = \tanh({z2[0,0]:.4f}) = {a2[0,0]:.4f}$")
                line3.set_text(rf"$a_2^{{(2)}} = \tanh({z2[1,0]:.4f}) = {a2[1,0]:.4f}$")
                line4.set_text(rf"$a^{{(2)}} = \left[{a2[0,0]:.4f},\ {a2[1,0]:.4f}\right]^T$")

            elif frame == 5:
                highlight_nodes(["a2_1", "a2_2", "yhat"])
                highlight_edges([("a2_1", "yhat"), ("a2_2", "yhat")])
                line1.set_text(r"$z^{(3)} = W^{(3)}a^{(2)} + b^{(3)}$")
                line2.set_text(
                    rf"$z^{{(3)}} = 1.1({a2[0,0]:.4f}) + (-0.6)({a2[1,0]:.4f}) + 0.02$"
                )
                line3.set_text(rf"$z^{{(3)}} = {z3[0,0]:.4f}$")
                line4.set_text(rf"$\hat{{y}} = z^{{(3)}} = {y_hat[0,0]:.4f}$")

            elif frame == 6:
                highlight_nodes(["yhat"])
                line1.set_text(r"$\mathcal{L} = (\hat{y} - y)^2$")
                line2.set_text(rf"$\mathcal{{L}} = ({y_hat[0,0]:.4f} - {y:.4f})^2$")
                line3.set_text(rf"$\mathcal{{L}} = {L[0,0]:.6f}$")
                line4.set_text(r"This completes the forward pass.")
                line5.set_text(r"Now we move backward through the network.")

            elif frame == 7:
                highlight_nodes(["yhat"])
                line1.set_text(r"Start backward pass from the loss.")
                line2.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial \hat{y}} = 2(\hat{y} - y)$")
                line3.set_text(rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial \hat{{y}}}} = {dL_dyhat[0,0]:.4f}$")
                line4.set_text(r"Because the output is linear,")
                line5.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial z^{(3)}} = \dfrac{\partial \mathcal{L}}{\partial \hat{y}}$")
                line6.set_text(rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial z^{{(3)}}}} = {dL_dz3[0,0]:.4f}$")

            elif frame == 8:
                highlight_nodes(["a2_1", "a2_2", "yhat"])
                highlight_edges([("a2_1", "yhat"), ("a2_2", "yhat")])
                line1.set_text(r"Output-layer gradients:")
                line2.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial W^{(3)}} = \dfrac{\partial \mathcal{L}}{\partial z^{(3)}} a^{(2)\top}$")
                line3.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial W^{{(3)}}}} = "
                    rf"\left[{dL_dW3[0,0]:.4f},\ {dL_dW3[0,1]:.4f}\right]$"
                )
                line4.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial b^{(3)}} = \dfrac{\partial \mathcal{L}}{\partial z^{(3)}}$")
                line5.set_text(rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial b^{{(3)}}}} = {dL_db3[0,0]:.4f}$")

            elif frame == 9:
                highlight_nodes(["a2_1", "a2_2", "z2_1", "z2_2"])
                highlight_edges([("z2_1", "a2_1"), ("z2_2", "a2_2")])
                line1.set_text(r"Backpropagate to the second hidden layer:")
                line2.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial a^{(2)}} = W^{(3)\top}\dfrac{\partial \mathcal{L}}{\partial z^{(3)}}$")
                line3.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial a^{{(2)}}}} = "
                    rf"\left[{dL_da2[0,0]:.4f},\ {dL_da2[1,0]:.4f}\right]^T$"
                )
                line4.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial z^{(2)}} = \dfrac{\partial \mathcal{L}}{\partial a^{(2)}} \odot (1-\tanh^2(z^{(2)}))$")
                line5.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial z^{{(2)}}}} = "
                    rf"\left[{dL_dz2[0,0]:.4f},\ {dL_dz2[1,0]:.4f}\right]^T$"
                )

            elif frame == 10:
                highlight_nodes(["a1_1", "a1_2", "z2_1", "z2_2"])
                highlight_edges([
                    ("a1_1", "z2_1"), ("a1_1", "z2_2"),
                    ("a1_2", "z2_1"), ("a1_2", "z2_2")
                ])
                line1.set_text(r"Gradients of the second hidden layer:")
                line2.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial W^{(2)}} = \dfrac{\partial \mathcal{L}}{\partial z^{(2)}} a^{(1)\top}$")
                line3.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial W^{{(2)}}}} = "
                    rf"\left[\left[{dL_dW2[0,0]:.4f},\, {dL_dW2[0,1]:.4f}\right],\,\left[{dL_dW2[1,0]:.4f},\, {dL_dW2[1,1]:.4f}\right]\right]$"
                )
                line4.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial b^{{(2)}}}} = "
                    rf"\left[{dL_db2[0,0]:.4f},\ {dL_db2[1,0]:.4f}\right]^T$"
                )
                line5.set_text(r"The second hidden layer mixes both features from the first hidden layer.")

            elif frame == 11:
                highlight_nodes(["a1_1", "a1_2", "z1_1", "z1_2"])
                highlight_edges([("z1_1", "a1_1"), ("z1_2", "a1_2")])
                line1.set_text(r"Backpropagate to the first hidden layer:")
                line2.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial a^{(1)}} = W^{(2)\top}\dfrac{\partial \mathcal{L}}{\partial z^{(2)}}$")
                line3.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial a^{{(1)}}}} = "
                    rf"\left[{dL_da1[0,0]:.4f},\ {dL_da1[1,0]:.4f}\right]^T$"
                )
                line4.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial z^{(1)}} = \dfrac{\partial \mathcal{L}}{\partial a^{(1)}} \odot (1-\tanh^2(z^{(1)}))$")
                line5.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial z^{{(1)}}}} = "
                    rf"\left[{dL_dz1[0,0]:.4f},\ {dL_dz1[1,0]:.4f}\right]^T$"
                )

            elif frame == 12:
                highlight_nodes(["x", "z1_1", "z1_2"])
                highlight_edges([("x", "z1_1"), ("x", "z1_2")])
                line1.set_text(r"Gradients of the first layer:")
                line2.set_text(r"$\dfrac{\partial \mathcal{L}}{\partial W^{(1)}} = \dfrac{\partial \mathcal{L}}{\partial z^{(1)}} x^\top$")
                line3.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial W^{{(1)}}}} = "
                    rf"\left[{dL_dW1[0,0]:.4f},\ {dL_dW1[1,0]:.4f}\right]^T$"
                )
                line4.set_text(
                    rf"$\dfrac{{\partial \mathcal{{L}}}}{{\partial b^{{(1)}}}} = "
                    rf"\left[{dL_db1[0,0]:.4f},\ {dL_db1[1,0]:.4f}\right]^T$"
                )
                line5.set_text(r"This completes backpropagation through all layers.")
                line6.set_text(r"We now have gradients for every trainable parameter.")

            elif frame == 13:
                highlight_nodes(list(pos.keys()))
                highlight_edges(edges)
                line1.set_text(r"Gradient descent update:")
                line2.set_text(r"$\theta \leftarrow \theta - \eta \nabla_{\theta}\mathcal{L}$")
                line3.set_text(
                    rf"$W^{{(1)}}_{{new}} = \left[{W1_new[0,0]:.4f},\ {W1_new[1,0]:.4f}\right]^T,\quad "
                    rf"b^{{(1)}}_{{new}} = \left[{b1_new[0,0]:.4f},\ {b1_new[1,0]:.4f}\right]^T$"
                )
                line4.set_text(
                    rf"$W^{{(2)}}_{{new}} = "
                    rf"\left[\left[{W2_new[0,0]:.4f},\, {W2_new[0,1]:.4f}\right],\,\left[{W2_new[1,0]:.4f},\, {W2_new[1,1]:.4f}\right]\right]$"
                )
                line5.set_text(
                    rf"$W^{{(3)}}_{{new}} = \left[{W3_new[0,0]:.4f},\ {W3_new[0,1]:.4f}\right],\quad "
                    rf"b^{{(3)}}_{{new}} = {b3_new[0,0]:.4f}$"
                )
                line6.set_text(rf"Learning rate: $\eta = {eta:.3f}$")

            elif frame == 14:
                highlight_nodes(list(pos.keys()))
                highlight_edges(edges)
                line1.set_text(r"New forward pass with updated parameters:")
                line2.set_text(
                    rf"Old prediction: $\hat{{y}} = {y_hat[0,0]:.4f}$"
                    rf" $\rightarrow$ New prediction: $\hat{{y}}_{{new}} = {y_hat_new[0,0]:.4f}$"
                )
                line3.set_text(
                    rf"Old loss: $\mathcal{{L}} = {L[0,0]:.6f}$"
                    rf" $\rightarrow$ New loss: $\mathcal{{L}}_{{new}} = {L_new[0,0]:.6f}$"
                )
                line4.set_text(
                    rf"$a^{{(1)}}_{{new}} = \left[{a1_new[0,0]:.4f},\ {a1_new[1,0]:.4f}\right]^T$"
                )
                line5.set_text(
                    rf"$a^{{(2)}}_{{new}} = \left[{a2_new[0,0]:.4f},\ {a2_new[1,0]:.4f}\right]^T$"
                )
                line6.set_text(r"The loss decreases because the parameters moved opposite to the gradient.")

            return (
                title_text, line1, line2, line3, line4, line5, line6,
                *[line for _, line in edge_artists],
                *[pt for pt, _ in node_artists.values()]
            )

        anim = FuncAnimation(
            fig,
            update,
            frames=len(stages),
            interval=interval,
            blit=False,
            repeat=repeat
        )

        plt.close(fig)
        return HTML(anim.to_jshtml())