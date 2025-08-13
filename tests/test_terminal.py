import os
import unittest
import warnings

from tethered_planning.utils import io
from tethered_planning.utils.colors import CmdColors

unittest.TestLoader.sortTestMethodsUsing = None  # run tests in order they are defined


class TestTerminalPrint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        abspath = os.path.abspath(__file__)
        dir_name = os.path.dirname(abspath)
        os.chdir(dir_name)
        if not os.path.exists("results"):
            os.makedirs("results")
        io.clean_folder("results")

    def test_terminal_color(self):
        print(f"{CmdColors.OKBLUE}[TEST]{CmdColors.ENDC} This is a status message.")
        print(
            f"{CmdColors.FAIL}[TEST]{CmdColors.ENDC} This is an error message that "
            "does not raise an exception nor end the program."
        )

    def test_warning(self):
        warnings.warn(
            f"{CmdColors.WARNING}[TEST]{CmdColors.ENDC} This is a warning message.",
            Warning,
        )


if __name__ == "__main__":
    unittest.main()
