PIPE_TXT = '\n\033[38;5;83mV-torch'
WHITE_TXT = '\033[38;5;15m:'
GOLDEN_TXT = '\033[38;5;222m:'
CLEAR_TXT = '\033[0;0m'

if __name__ == '__main__':
    print(PIPE_TXT + f'{WHITE_TXT} Example{CLEAR_TXT}')
    print(PIPE_TXT + f'{GOLDEN_TXT} Example{CLEAR_TXT}')