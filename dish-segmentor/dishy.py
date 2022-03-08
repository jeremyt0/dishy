from analysis.auto import Automator
import argparse
import sys
import os


class Main:
    def main(path):
        print(f"Project directory: {path}")

        images_dir = path
        if path == "":
            root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            images_dir = os.path.join(root_dir, "IMAGES")
            os.makedirs(images_dir, exist_ok=True)

        program = Automator(project_dir=images_dir)

        program.run()



if __name__=="__main__":
    print("### Start ###")
    PARSER = argparse.ArgumentParser(
        description="""Generates CSV results of colour information from leaf-disc images.

        Description: This script segments the leaf discs into blocks and analyses the colour of each leaf.

        """,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    PARSER.add_argument("--path", type=str, required=True, default=".", help="Folder path of images")

    # Show help if no arguments
    if len(sys.argv)==1:
        PARSER.print_help(sys.stderr)
        sys.exit(1)

    ARGS = PARSER.parse_args()

    Main.main(ARGS.path)

    print("### End ###", file=sys.stderr)