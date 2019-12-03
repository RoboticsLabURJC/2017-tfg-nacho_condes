import os
from time import sleep


# Parameters
model = input('Model filename >> ')
input_w = int(input('Input width >> '))
input_h = int(input('Input height >> '))



precisions = ['FP32', 'FP16']
msss = [3, 20, 50]
mces = [1, 3, 5]

allow_growth = True

for precision in precisions:
    for mss in msss:
        for mce in mces:
            command = f'python trter.py {model} {input_w} {input_h}' +
                            f'{precision} {mss} {mce} {allow_growth}'
            os.system(command)
            sleep(2)

