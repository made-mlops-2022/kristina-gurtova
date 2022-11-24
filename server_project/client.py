import argparse
import requests

parser = argparse.ArgumentParser(
    prog='Client',
    description='Program is simular to curl')
parser.add_argument('url')  # positional argument
parser.add_argument('-t', '--type', default="GET", choices=["GET", "POST"])  # option that takes a value
parser.add_argument('-d', '--data', required=False)  # on/off flag

if __name__ == "__main__":
    args = parser.parse_args()
    if args.type == "GET":
        resp = requests.get(args.url)
    else:
        resp = requests.post(args.url, data=args.data)
    print(resp.content.decode("utf-8"))
