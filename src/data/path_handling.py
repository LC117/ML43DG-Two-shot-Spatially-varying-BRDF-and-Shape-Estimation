from pathlib import Path


class PathHandler:
    def __init__(self) -> None:
        """
        Here all the paths that are relevant will be added:
        """
        self.root_dir = Path(__file__).resolve().parent.parent # PosixPath('.../ML43DG-Two-shot-Spatially-varying-BRDF-and-Shape-Estimation/src')
        self.data_dir = self.check_path(self.root_dir / "data")
        self.model_dir = self.check_path(self.root_dir / "model")
    
    
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
        if path.exists():
            return path
        raise Exception(f"Path: \"{str(path)}\" does not exist!")
    

# This object will be imported from others and used as entry to get all paths!
path_manager = PathHandler()
