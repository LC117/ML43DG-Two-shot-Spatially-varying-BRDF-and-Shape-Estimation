from pathlib import Path


class PathHandler:
    
    root_dir = Path(__file__).resolve().parent
    
    def __init__(self) -> None:
        """
        Here all the paths that are relevant will be added:
        """
        data_dir = self.check_path(self.root_dir / "data")
        model_dir = self.check_path(self.root_dir / "model")
        notebooks_dir = self.check_path(self.root_dir / "notebooks")
    
    
    @staticmethod
    def check_path(path: Path):
        """Checks for path validity.

        Args:
            path (Path): path to check

        Raises:
            Exception: Path does not exist.

        Returns:
            path (Path): input path
        """
        if path.exist():
            return path
        raise Exception(f"Path: \"{str(path)}\" does not exist!")
    

# This object will be imported from others and used as entry to get all paths!
path_manager = PathHandler()
