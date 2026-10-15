"""Brain Signatures Lab banner, printed when the pipeline starts.

The logo is a palette-indexed pixel grid (b = blue, r = red, n = navy, . = empty) drawn two pixel
rows per terminal line with half-block characters. Without a colour terminal, or with NO_COLOR set,
a plain-text header is printed instead.
"""
import os
import sys

LAB_NAME = 'Brain Signatures Lab'
LAB_URL = 'https://www.brainsigns.net/'

PALETTE = {'b': (16, 112, 232), 'r': (226, 59, 59), 'n': (63, 81, 181)}

LOGO = (
    '.......bbbbbbb',
    '.....bbbbbbbbbbb',
    '...bbb.........bbb',
    '..bb.............bb...nnnnnnnn....nnnnnnnn........nn.......nn...nnn....nn',
    '..b..bbbbb.bbbb...bb..nn.....nn...nn.....nn......nnnn......nn...nnnn...nn',
    '.bb.bb...bbb..bb...b..nn.....nn...nn.....nn.....nn..nn.....nn...nn.nn..nn',
    'bb..b...rr.rr..bb..bb.nnnnnnnn....nnnnnnnn.....nn....nn....nn...nn..nn.nn',
    'bb..b..rrrrrrr.bb..bb.nn.....nn...nn...nn.....nn......nn...nn...nn...nnnn',
    'bb..b...rrrrr..bb...b.nn.....nn...nn....nn....nnnnnnnnnn...nn...nn....nnn',
    'bb..bb...rrr...b....b.nn.....nn...nn.....nn...nn......nn...nn...nn.....nn',
    'bb...bb...r...bb...bb.nnnnnnnn....nn.....nn...nn......nn...nn...nn.....nn',
    '.bb...b......bb....b',
    '..bb...bb...bb...bbb',
    '...bb...bbbb..bbbb',
    '....bbb...bbbbbb',
    '.....bbbb...bb',
    '.......bbbb.bb',
    '..........bbbb',
    '............bb',
    '............bb',
)

# Terminal lines (two logo rows each) that carry text, and the column the text starts at.
TEXT_COLUMN = 22
TEXT_LINES = {6: ('n', 'S  I  G  N  A  T  U  R  E  S     L  A  B'),
              8: ('b', LAB_URL)}

ESC = '\033['
RESET = ESC + '0m'
BOLD = ESC + '1m'
UPPER, LOWER, FULL = '▀', '▄', '█'


def _fg(shade):
    return ESC + '38;2;{};{};{}m'.format(*PALETTE[shade])


def _bg(shade):
    return ESC + '48;2;{};{};{}m'.format(*PALETTE[shade])


def _logo_lines():
    width = max(map(len, LOGO))
    grid = [row.ljust(width, '.') for row in LOGO]
    if len(grid) % 2:
        grid.append('.' * width)
    lines = []
    for top, bottom in zip(grid[0::2], grid[1::2]):
        cells = []
        for upper, lower in zip(top, bottom):
            if upper == '.' and lower == '.':
                cells.append(' ')
            elif upper == lower:
                cells.append(_fg(upper) + FULL + RESET)
            elif lower == '.':
                cells.append(_fg(upper) + UPPER + RESET)
            elif upper == '.':
                cells.append(_fg(lower) + LOWER + RESET)
            else:
                cells.append(_fg(upper) + _bg(lower) + UPPER + RESET)
        while cells and cells[-1] == ' ':
            cells.pop()
        lines.append(cells)
    return lines


def use_colour(stream):
    return (hasattr(stream, 'isatty') and stream.isatty() and 'NO_COLOR' not in os.environ
            and os.environ.get('TERM') != 'dumb')


def banner(stream=sys.stdout):
    if not use_colour(stream):
        return f'{LAB_NAME}  |  {LAB_URL}\n'
    lines = _logo_lines()
    for index, (shade, text) in TEXT_LINES.items():
        lines[index] += [' '] * max(TEXT_COLUMN - len(lines[index]), 1) + [BOLD + _fg(shade) + text + RESET]
    return '\n' + '\n'.join('  ' + ''.join(cells) for cells in lines) + '\n'


def print_banner(stream=sys.stdout):
    try:
        stream.write(banner(stream) + '\n')
    except UnicodeEncodeError:
        stream.write(f'{LAB_NAME}  |  {LAB_URL}\n\n')
    stream.flush()
