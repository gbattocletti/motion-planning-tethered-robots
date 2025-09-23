def combine_colors(c1: str, c2: str, w1: float = 0.5, w2: float = 0.5) -> str:
    """
    Combine two hex RGB colors by averaging their RGB components.

    Args:
        c1 (str): First color as a hex RGB string
        c2 (str): Second color as a hex RGB string

        w1 (float, optional): Weight of the first color. Defaults to 0.5.
        w2 (float, optional): Weight of the second color. Defaults to 0.5.

    Returns:
        str: Combined color as a hex RGB string

    Raises:
        TypeError: If the input colors are not strings
        ValueError: If the input colors are not valid hex RGB strings
        ValueError: If the weights are not between 0 and 1
    """

    # Helper function to zero-pad hex values
    def zpad(x):
        if len(x) == 2:
            return x
        else:
            return "0" + x

    # Validate inputs
    if not (isinstance(c1, str) and isinstance(c2, str)):
        raise TypeError("Colors must be hex RGB strings")
    if not (c1.startswith("#") and c2.startswith("#")):
        raise ValueError("Colors must start with '#'")
    if not (len(c1) == 7 and len(c2) == 7):
        raise ValueError("Colors must be 7 characters long (e.g. '#RRGGBB')")
    if not (0 <= w1 <= 1 and 0 <= w2 <= 1):
        raise ValueError("Weights must be between 0 and 1")

    # Compute weighted average of RGB components
    red = int(int(c1[1:3], 16) * w1 + int(c2[1:3], 16) * w2)
    green = int(int(c1[3:5], 16) * w1 + int(c2[3:5], 16) * w2)
    blue = int(int(c1[5:7], 16) * w1 + int(c2[5:7], 16) * w2)

    # Convert back to hex RGB string
    # [2:] to remove '0x' prefix from hex conversion, "#" to form hex color string
    c_mix = "#" + zpad(hex(red)[2:]) + zpad(hex(green)[2:]) + zpad(hex(blue)[2:])
    return c_mix


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
