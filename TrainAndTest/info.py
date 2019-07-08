import sys
import General.launcher as G

def main():
    if len(sys.argv) == 1:
        print ("Missing path to config file. Exit.")
        return
    G.parse_config_info(sys.argv[1])

if __name__ == "__main__":
    main()