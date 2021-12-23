as_train        = "../Data/Training/as_training_simple.utf8"
cityu_train     = "../Data/Training/cityu_training_simple.utf8"
msr_train       = "../Data/Training/msr_training_simple.utf8"
pku_train       = "../Data/Training/pku_training_simple.utf8"

out_train       = "../Data/Training/concat_training.utf8"

as_dev          = "../Data/Dev/as_test_simple_gold.utf8"
cityu_dev       = "../Data/Dev/cityu_test_simple_gold.utf8"
msr_dev         = "../Data/Dev/msr_test_simple_gold.utf8"
pku_dev         = "../Data/Dev/pku_test_simple_gold.utf8"

out_dev         = "../Data/Dev/concat_dev.utf8"

filenames_train = [as_train, cityu_train, msr_train, pku_train]
filenames_dev   = [as_dev, cityu_dev, msr_dev, pku_dev]

with open(out_train, 'w') as outfile:
    for fname in filenames_train:
        with open(fname) as infile:
            outfile.write(infile.read())

with open(out_dev, 'w') as outfile:
    for fname in filenames_dev:
        with open(fname) as infile:
            outfile.write(infile.read())

