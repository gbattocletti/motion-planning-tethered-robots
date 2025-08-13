class CmdColors:
    """
    Class of ANSI escape sequences to print colored output to the terminal.
    """

    # For more details see:
    # https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal  # pylint: disable=line-too-long
    # ANSI escape sequences color list:
    # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class CustomColors:
    """
    Custom color palette for the simulations visualization.
    """

    # Colors are defined as [R G B] lists (note: values must be between 0 and 1)
    # https://matplotlib.org/stable/tutorials/colors/colors.html

    # Base colors
    white = [1, 1, 1]
    black = [0, 0, 0]
    darker_gray = [0.502, 0.502, 0.502]
    dark_gray = [0.663, 0.663, 0.663]
    light_gray = [0.827, 0.827, 0.827]
    lighter_gray = [0.941, 0.941, 0.941]

    # Custom colors
    yellow = [0.866, 0.643, 0.282]
    blue1 = [0.282, 0.643, 0.866]
    blue2 = [0.106, 0.149, 0.31]
    blue3 = [0.431, 0.706, 0.82]
    red = [0.635, 0.243, 0.282]


class PlotColors(CustomColors):
    """
    Determines the assignment of colors to the simulations objects to be displayed in
    the plots. Plot alpha values, markers type, and line thickess are defined directly
    in the plot functions.
    """

    # obstacles
    obstacles_color = CustomColors.darker_gray
    obstacles_edges_color = obstacles_color

    # generators
    generators_color = CustomColors.dark_gray

    # goal
    goal_color = CustomColors.yellow
    goal_edge_color = goal_color

    # robot
    robot_color = CustomColors.blue3
    robot_edge_color = robot_color

    # anchor
    anchor_color = CustomColors.red
    anchor_edge_color = anchor_color

    # tether
    tether_color = CustomColors.black

    # graph
    node_color = CustomColors.blue2
    edge_color = CustomColors.blue1

    # other curves
    other_curves_cmap = [
        "#000033",
        "#000066",
        "#000099",
        "#0000CC",
        "#0000FF",
        "#003366",
        "#003399",
        "#0033CC",
        "#0033FF",
        "#006699",
    ]
    other_curves_n = len(other_curves_cmap)
