from dishy import Main
from utilities.logg import LOGGER

if __name__=="__main__":
    LOGGER.debug("Starting application...")
    # path = "../images"  # Mac
    path = ""
    Main.main(path=path)
    LOGGER.success("End of program.")
