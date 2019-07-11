import sys
import General.launcher as launcher


def main():
    if len(sys.argv) == 1:
        print ("Missing path to config file. Exit.")
        return
    launcher.parse_config(sys.argv[1])
    if len(sys.argv) > 2:
        launcher.parse_request(sys.argv[2])
    else:
        launcher.parse_request("");
    launcher.work()


if __name__ == "__main__":
    main()

