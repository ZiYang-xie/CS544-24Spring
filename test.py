import os
import argparse
from testDir.subTest import MYDICT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS544-24Spring/mp3')
    parser.add_argument('--num_vars', type=int, default=100, help='Number of variables')
    parser.parse_args()  # Parse the arguments
    print(MYDICT)

# import os
# import argparse

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='CS544-24Spring/mp3')
#     parser.add_argument('--num_vars', type=int, default=100, help='Number of variables')
#     args = parser.parse_args()  # Parse the arguments

#     # Now you can use args.num_vars in your script
#     print(f"Number of variables: {args.num_vars}")
