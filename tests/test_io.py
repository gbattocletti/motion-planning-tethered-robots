import os
import unittest

from tethered_planning.env.env_2d import Env2D
from tethered_planning.utils import io, plot
from tethered_planning.utils.colors import CmdColors
from tethered_planning.utils.settings import Settings

unittest.TestLoader.sortTestMethodsUsing = None  # run tests in order they are defined


class TestIO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        abspath = os.path.abspath(__file__)
        dir_name = os.path.dirname(abspath)
        os.chdir(dir_name)
        if not os.path.exists("results"):
            os.makedirs("results")
        io.clean_folder("results")

    def test_io_folder_creation(self):
        print(
            f"{CmdColors.OKBLUE}[TestIO]{CmdColors.ENDC} Running "
            "test_io_folder_creation."
        )
        io.create_io_folders()
        sim_id, sim_name = io.create_sim_folder()
        sim_folder = f"./results/{sim_name}"
        print(
            f"{CmdColors.OKBLUE}[TestIO]{CmdColors.ENDC}\n"
            f"\tSim folder: {sim_folder}\n"
            f"\tSim ID: {sim_id}\n"
            f"\tSim name: {sim_name}"
        )

    def test_load_yaml(self):
        print(f"{CmdColors.OKBLUE}[TestIO]{CmdColors.ENDC} Running test_load_yaml.")
        data = io.load_yaml("data/test_yaml.yaml")  # load_yaml requires full path
        self.assertIsInstance(data, dict)
        for key, value in data.items():
            print(f"{key}: {value}")

    def test_load_yaml_empty(self):
        print(
            f"{CmdColors.OKBLUE}[TestIO]{CmdColors.ENDC} Running "
            "test_load_yaml_empty."
        )
        data = io.load_yaml("data/test_yaml_empty.yaml")
        self.assertIsInstance(data, dict)
        for key, value in data.items():
            print(f"{key}: {value}")

    def test_write_readme(self):
        print(f"{CmdColors.OKBLUE}[TestIO]{CmdColors.ENDC} Running test_write_readme.")
        settings = Settings()
        env = Env2D(settings)
        io.write_readme(settings, env)

    def test_log_settings(self):
        print(f"{CmdColors.OKBLUE}[TestIO]{CmdColors.ENDC} Running test_log_settings.")
        settings = Settings()
        io.log_sim_data(settings)

    def test_save_figure(self):
        print(f"{CmdColors.OKBLUE}[TestIO]{CmdColors.ENDC} Running test_save_figure.")
        settings = Settings("test_settings")
        settings.env_name = "test_env_4.yaml"
        env = Env2D(settings)
        settings.anim.animate = True
        settings.plot.show_legend = True
        fig, _ = plot.plot_env(env, settings)
        io.save_figure(fig, settings, "test_save_figure")
        io.save_figure(fig, settings, "test_save_figure", "pdf")


if __name__ == "__main__":
    unittest.main()
