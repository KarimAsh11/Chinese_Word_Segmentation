import argparse
from utility import read, write


parser = argparse.ArgumentParser(description='Dataset preparation')


parser.add_argument('raw_input',
                    help='Raw input file')

parser.add_argument('input',
                    help='Input file')

parser.add_argument('labels',
                    help='Label file')


def run(raw_input, input, labels):
    """
    Read input, create copy and label files following the BIO encoding.

    params:
        input (str): input file path
        copy (str): copy file path with no spaces
        labels (str): labels file path containing BIO encoding
    """
    labels_output = []
    copy_output = []

    input_data = read(raw_input)
    for line in input_data.split('\n'):
        words = line.split()
        copy_output.append(words)      
        curr_encoding = []

        for word in words:
            if (len(word) == 1):
                curr_encoding.append('S')
            elif (len(word) == 2):
                curr_encoding.append("BE")
            elif (len(word) > 2):
                curr_encoding.append("B")

                for w in range(len(word)-2):
                    curr_encoding.append("I")
  
                curr_encoding.append("E")

        labels_output.append(curr_encoding)

    write(copy_output, input)
    write(labels_output, labels)


if __name__ == '__main__':
    args = parser.parse_args()
    run(**vars(args))
