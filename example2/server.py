from multifit import EVBServer
import sys
import logging


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    fh = logging.FileHandler("%s.log"%sys.argv[1])
    fh.setLevel(logging.WARNING)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    server = EVBServer(int(sys.argv[1]))
    server.listen()
