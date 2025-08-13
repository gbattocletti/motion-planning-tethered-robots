import os
import unittest

from tethered_planning.utils import io
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

    def test_settings_default(self):
        print(
            f"{CmdColors.OKBLUE}[TestSettings]{CmdColors.ENDC} Running "
            "test_settings_default."
        )
        settings = Settings()
        print(settings)

    def test_load_settings_file(self):
        print(
            f"{CmdColors.OKBLUE}[TestSettings]{CmdColors.ENDC} Running "
            "test_load_settings_file."
        )
        settings = Settings("test_settings")
        print(settings)

    def test_load_settings_custom(self):
        print(
            f"{CmdColors.OKBLUE}[TestSettings]{CmdColors.ENDC} Running "
            "test_load_settings_custom."
        )
        settings = Settings()
        print(settings)
        settings.load_settings_custom("test_settings")
        print(settings)


if __name__ == "__main__":
    unittest.main()
