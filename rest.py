import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def main():
    # Initial parameter value for a
    a0 = 1.0

    # Create the figure and axis for the plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Leave space at the bottom for the slider

    # Generate x values and compute the initial y values
    x = np.linspace(-10, 10, 400)
    y = -x**2 + a0 * x

    # Plot the function
    line, = ax.plot(x, y, label=f'$-x^2 + {a0}x$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Plot of $-x^2 + a x$')
    ax.legend()
    ax.grid(True)

    # Define the slider axis: [left, bottom, width, height]
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_a = Slider(ax=ax_slider, label='a', valmin=-10, valmax=10, valinit=a0)

    # Update function to redraw the plot when slider value changes
    def update(val):
        a = slider_a.val
        y = -x**2 + a * x
        line.set_ydata(y)
        line.set_label(f'$-x^2 + {a:.2f}x$')
        ax.legend()  # Update legend to show the new parameter value
        fig.canvas.draw_idle()

    # Register the update function with the slider
    slider_a.on_changed(update)

    plt.show()

if __name__ == '__main__':
    main()