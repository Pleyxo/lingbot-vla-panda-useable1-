import numpy as np
import os
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import array_to_string

# Path to the XML file relative to this script
_ARENA_XML_DIR = os.path.dirname(os.path.abspath(__file__))


class TableBinArena(Arena):
    """
    Workspace that contains a table with an open-top bin (target box) on it.

    The bin is an open box placed on the table surface. The robot must
    pick up the cube and place it into this bin.

    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction
        bin_pos (3-tuple): (x,y,z) position of the target bin center
    """

    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        bin_pos=(0.15, 0.15, 0.82),
    ):
        xml_path = os.path.join(_ARENA_XML_DIR, "assets/arenas", "table_bin_arena.xml")
        super().__init__(xml_path)

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction

        self.bin_pos = np.array(bin_pos)
        self.bin_body = self.worldbody.find("./body[@name='target_bin']")

        # Bin boundaries for "in bin" checking
        # bin half-size is 0.08 x 0.08, wall height is 0.04 from bottom
        self.bin_half_size_xy = 0.075  # inner half-size (slightly smaller than outer)
        self.bin_bottom_z = bin_pos[2]  # z of bin bottom
        self.bin_top_z = bin_pos[2] + 0.08  # z of bin top (bottom + wall height * 2)

        self.configure_location()

    def configure_location(self):
        """Configures correct locations for this arena"""
        self.floor.set("pos", array_to_string(self.bottom_pos))
