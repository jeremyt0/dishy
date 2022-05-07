from dishy import Main
from utilities.logg import LOGGER

import os

if __name__=="__main__":
    try:
        LOGGER.debug("Starting application...")
        # path = "../images"  # Mac
        path = ""
        Main.main(path=path)
        LOGGER.success("End of program.")
    except Exception as e:
        print(e)
    os.system('pause')
