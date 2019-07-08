import sys
import General.launcher as G

def main():
    if len(sys.argv) == 1:
        print ("Missing path to config file. Exit.")
        return
    G.parse_config(sys.argv[1])
    G.parse_request(sys.argv[2])
    G.work()

if __name__ == "__main__":
    main()
