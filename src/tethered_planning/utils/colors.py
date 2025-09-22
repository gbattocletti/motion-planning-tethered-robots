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

    # Colors are defined as hex RGB lists so that VSCode shows a preview.
    # https://matplotlib.org/stable/tutorials/colors/colors.html

    # Base colors
    white = "#FFFFFF"
    black = "#000000"
    darker_gray = "#808080"
    dark_gray = "#A9A9A9"
    light_gray = "#D0D0D0"
    lighter_gray = "#F0F0F0"

    # Custom colors
    yellow = "#DFAF45"
    yellow1 = "#B49900"
    yellow2 = "#DAC892"
    blue = "#0080FF"
    blue1 = "#1A2A4D"
    blue2 = "#48A1B0"
    red = "#CC3F3F"
    red1 = "#8B0000"
    red2 = "#C98383"

    # Colormaps
    blue_cmap = [
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

    # colormap for homotopy-agumented triangulation. 10 colors are passed. The list
    # needs to be sliced to match the number of unique signatures in the triangulation.
    # If more than 10 signatures are present, the list must be extended.
    layers_cmap = [
        "#800000",
        "#C54444",
        "#D66969",
        "#C47D21",
        "#C9AA69",
        "#FFEB3B",
        "#BCD857",
        "#11B92D",
        "#3CFF4C",
        "#80FA8A",
    ]


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
    robot_color = CustomColors.blue
    robot_edge_color = robot_color

    # anchor
    anchor_color = CustomColors.red
    anchor_edge_color = anchor_color

    # tether
    tether_color = CustomColors.black

    # graph
    node_color = CustomColors.blue1
    edge_color = CustomColors.blue2
    node_dual_color = CustomColors.red1
    edge_dual_color = CustomColors.red2

    # points, curves, polygons
    points_color = CustomColors.blue
    curves_cmap = CustomColors.blue_cmap
    curves_n = len(curves_cmap)
    polygons_color = CustomColors.blue
