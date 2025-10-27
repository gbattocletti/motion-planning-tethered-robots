import datetime
import os
from pathlib import Path

from typing_extensions import override

from tethered_planning.utils import io
from tethered_planning.utils.colors import CmdColors


class SettingsBase:
    """
    Base class for all settings. Contains common methods for loading and settings, and
    to list the available settings.
    """

    def __init__(self) -> None:
        """
        Initializes the SettingsBase class.
        """
        self.__available_settings__: list = []  # available attributes

    def get_available_settings(self) -> list:
        """
        Return a list of the attributes of a class, excluding the dunder methods and the
        callable methods.

        Returns:
            available_settings (list): a list of the attributes of the class.
        """
        # Build list of available settings
        available_settings: list = [
            attr
            for attr in dir(self)
            if not attr.startswith("__") and not callable(getattr(self, attr))
        ]

        # Return the list
        return available_settings

    def export_settings(self) -> None:
        """
        Export the list of class attributes to a .yaml file.

        Returns:
            None
        """
        # Collect attributes
        attributes: list = self.get_available_settings()

        # Create YAML file and write settings
        filename = f"settings_list_{type(self).__name__}.yaml"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Settings Class: {type(self).__name__}\n")
            f.write("Available Settings:\n")
            for attr in attributes:
                f.write(f'\t"{attr}:"\n')

    def load_settings_default(self, settings_dict: dict) -> None:
        """
        Load the default settings from the settings dictionary loaded from the
        settings_default.yaml file.

        Args:
            settings_dict (dict): a dictionary containing the default settings.

        Raises:
            KeyError: if a required setting is not found in the dictionary.
            TypeError: if settings_dict is not a dictionary.

        Returns:
            None
        """
        # check input type
        if not isinstance(settings_dict, dict):
            print(
                f"{CmdColors.FAIL}[Settings]{CmdColors.ENDC} The settings must be "
                "passed as a dictionary."
            )
            raise TypeError

        # load the settings
        for key in self.__available_settings__:
            try:
                setattr(self, key, settings_dict[key])
            except KeyError as e:
                print(
                    f"{CmdColors.FAIL}[Settings]{CmdColors.ENDC} KeyError: "
                    f"{key} not found in the default settings file."
                )
                raise KeyError from e

    def load_settings_dict(self, settings_dict: dict) -> None:
        """
        Load the custom settings from the dictionary passed as argument.

        Args:
            settings_dict (dict): a dictionary containing the settings.

        Raises:
            KeyError: if a required setting is not found in the dictionary.
            TypeError: if settings_dict is not a dictionary.

        Returns:
            None
        """
        # Check input type
        if not isinstance(settings_dict, dict):
            print(
                f"{CmdColors.FAIL}[Settings]{CmdColors.ENDC} The settings must be "
                "passed as a dictionary."
            )
            raise TypeError

        # Load the settings
        for key, value in settings_dict.items():
            if key not in self.__available_settings__:
                print(
                    f"{CmdColors.WARNING}[Settings]{CmdColors.ENDC} {key} is not an "
                    "available setting."
                )
            else:
                setattr(self, key, value)


class Settings(SettingsBase):
    """
    Settings class with the main settings for the simulation.
    """

    def __init__(
        self,
        settings_filename: str = None,
        create_sim_folder: bool = False,
    ) -> None:
        """
        Initializes the Settings class.

        Args:
            settings_filename (str, optional): Custom settings filename.

        Kwargs:
            create_sim_folder (bool, optional): If True, creates a new folder to save
                the simulation data. If false, only generates the simulation name and
                stores the results directly in the results folder. Default is False.

        Returns:
            None
        """
        super().__init__()

        # Initialize Settings class attributes
        self.__settings_filename__: str = None
        self.__start_datetime__: datetime.datetime = datetime.datetime.now()
        self.__end_datetime__: datetime.datetime = None
        self.__start_date__: str = self.__start_datetime__.strftime("%Y-%m-%d")
        self.__end_date__: str = None
        self.__start_time__: str = self.__start_datetime__.strftime("%H:%M:%S")
        self.__end_time__: str = None
        self.__elapsed_time__: str = None
        self.sim_folder: str = None  # generated at runtime
        self.sim_id: str = None  # generated at runtime
        self.sim_name: str = None  # generated at runtime
        self.fix_seed: bool = None
        self.seed: int = None
        self.env_name: str = None
        self.planner: SettingsPlanner = SettingsPlanner()

        # Get list of available settings attributes
        self.__available_settings__ = self.get_available_settings()

        # Load the default settings
        self.load_settings_default()

        # If a custom settings file is provided, load the settings from the file,
        # overwriting (part of) the default settings.
        if settings_filename is not None:
            if settings_filename.endswith(".yaml"):
                settings_filename = settings_filename[:-5]  # remove the .yaml extension
            self.load_settings_custom(settings_filename)
        else:
            print(
                f"\n{CmdColors.WARNING}[Settings]{CmdColors.ENDC} No settings file "
                "provided. Using default settings."
            )

        # Check kwargs
        if not isinstance(create_sim_folder, bool):
            print(
                f"{CmdColors.FAIL}[Settings]{CmdColors.ENDC} The argument "
                "create_sim_folder must be a boolean."
            )
            raise TypeError

        # Generate the sim name and create the results folder
        io.create_io_folders()
        if create_sim_folder is True:
            self.sim_id, self.sim_name = io.create_sim_folder()
        else:
            self.sim_id, self.sim_name = io.create_sim_name()
        self.sim_folder = f"results/{self.sim_name}"

    @override
    def load_settings_default(self, _=None) -> None:
        """
        Loads the default settings by reading the settings_default.yaml file.

        Raises:
            KeyError: If a required setting is missing from the default settings file.

        Returns:
            None
        """

        # Set the default settings file name
        self.__settings_filename__ = "settings_default"

        # Compose the path to the default settings file
        default_settings_dict: dict = io.load_yaml(
            os.path.join(Path(__file__).parent, "settings_default.yaml")
        )

        # Load the settings
        for key in self.__available_settings__:
            if key == "planner":
                self.planner.load_settings_default(default_settings_dict["planner"])
            else:
                try:
                    setattr(self, key, default_settings_dict[key])
                except KeyError as e:
                    print(
                        f"{CmdColors.FAIL}[Settings]{CmdColors.ENDC} KeyError: "
                        f"{key} not found in the default settings file (required)."
                    )
                    raise KeyError from e

    @override
    def load_settings_dict(self, _=None) -> None:
        print(
            f"{CmdColors.WARNING}[Settings]{CmdColors.ENDC} The function "
            "load_settings_dict is not implemented for the class Settings. Please use "
            "the load_settings_custom method instead."
        )

    def load_settings_custom(self, settings_filename: str) -> None:

        # save the name of the settings file being loaded
        self.__settings_filename__ = settings_filename

        # load the settings from the YAML file
        settings_file_path = f"data/{settings_filename}.yaml"
        settings_dict: dict = io.load_yaml(settings_file_path)

        # load the settings from the file
        for key, value in settings_dict.items():
            if key not in self.__available_settings__:
                print(
                    f"{CmdColors.WARNING}[Settings]{CmdColors.ENDC} {key} is not an "
                    f"available setting. Please check the file {settings_file_path} "
                    "for typos, and the class Settings for the available settings."
                )
            elif key in ("data_folder", "results_folder"):
                print(
                    f"{CmdColors.WARNING}[Settings]{CmdColors.ENDC} {key} is a "
                    "reserved setting and cannot be changed. Using default values."
                )
            elif key == "planner":
                self.planner.load_settings_dict(value)
            else:
                setattr(self, key, value)

        # print completion message
        print(
            f"{CmdColors.OKBLUE}[Settings]{CmdColors.ENDC} Settings were "
            f"successfully loaded from the file {settings_filename}."
        )

    def elapsed_time(self) -> None:
        """
        Computes simulation elapsed time.
        """
        self.__end_datetime__ = datetime.datetime.now()
        self.__end_date__ = self.__end_datetime__.strftime("%Y-%m-%d")
        self.__end_time__ = self.__end_datetime__.strftime("%H:%M:%S")
        self.__elapsed_time__ = (
            self.__end_datetime__ - self.__start_datetime__
        ).strftime("%H:%M:%S")
        elapsed_time: datetime.timedelta = self.__end_time__ - self.__start_datetime__
        print(
            f"{CmdColors.OKBLUE}[Settings]{CmdColors.ENDC} Elapsed time: {elapsed_time}"
        )

    def __str__(self) -> str:
        return (
            f"{CmdColors.OKBLUE}[Settings]{CmdColors.ENDC} Settings file: "
            f"data/{self.__settings_filename__}.yaml"
        )


class SettingsPlanner(SettingsBase):
    """
    Settings class with the planner settings.
    """

    def __init__(self) -> None:
        super().__init__()

        # General settings
        self.name: str = None

        # RRT settings
        self.max_edge_length: float = None
        self.goal_bias: bool = None
        self.goal_bias_rate: float = None
        self.goal_reached_termination_condition: bool = None
        self.max_nodes_n_termination_condition: bool = None
        self.max_nodes_n: int = None

        # Triangulation settings
        self.search_algorithm: str = None

        self.__available_settings__ = self.get_available_settings()
