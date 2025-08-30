# Sudoku Helper

A lightweight Python tool to assist in solving Sudoku puzzles.  
This project provides common Sudoku solving strategies and utilities for both beginners and advanced players.
---
## Features
- Supports **standard 9x9 Sudoku** puzzles.
- Implements multiple solving techniques:
- Step-by-step solving process with explanations.

###### Supported techniques:
- Hidden Single
- Naked Single
- Locked Candidate
- Hidden Pair
- Hidden Triple
- Hidden Quad
- Naked Pair
- Naked Triple
- Naked Quad
- X-Wing
- Swordfish
- Jellyfish
- Color Trap
- Color Wrap
- Color Wing
###### Future implementations:
- XY-Wing
- XYZ-Wing
- XY-Chain
- Finned X-Wing
- Finned Swordfish
- Finned Jellyfish
- Empty Rectangle
- Aligned Pair Exclusion
- Almost Locked Set
- Unique Rectangle
---

## Installation

Clone the repository:
```bash
git clone https://github.com/Galaxy-rio/sudoku_solver.git
cd sudoku-helper
```
---

## Usage

### Run sudoku helper
```bash
python sudoku_helper.py
```

### Example

```bash
input sudoku (9 lines, use 0 for empty cells):
714060930
068790420
000040076
000170043
140835207
073020000
057910382
001080754
002057619

output:
当前候选棋盘：
      |       |       | ||   2   |       |   2   | ||       |       |       |
  7   |   1   |   4   | ||   5   |   6   |       | ||   9   |   3   |   5   |
      |       |       | ||       |       |   8   | ||       |       |   8   |
    3 |       |       | ||       |       | 1   3 | ||       |       | 1     |
  5   |   6   |   8   | ||   7   |   9   |       | ||   4   |   2   |   5   |
      |       |       | ||       |       |       | ||       |       |       |
  2 3 |   2 3 |       | ||   2 3 |       | 1 2 3 | || 1     |       |       |
  5   |       |   5   | ||   5   |   4   |       | ||   5   |   7   |   6   |
    9 |     9 |     9 | ||       |       |   8   | ||   8   |       |       |
------------------------------------------------------------------------------
  2   |   2   |       | ||       |       |       | ||       |       |       |
  5 6 |       |   5 6 | ||   1   |   7   |     6 | ||   5   |   4   |   3   |
  8 9 |   8 9 |     9 | ||       |       |     9 | ||   8   |       |       |
      |       |       | ||       |       |       | ||       |       |       |
  1   |   4   |     6 | ||   8   |   3   |   5   | ||   2   |     6 |   7   |
      |       |     9 | ||       |       |       | ||       |     9 |       |
      |       |       | ||       |       |       | || 1     |       | 1     |
  5 6 |   7   |   3   | || 4   6 |   2   | 4   6 | ||   5   |     6 |   5   |
  8 9 |       |       | ||       |       |     9 | ||   8   |     9 |   8   |
------------------------------------------------------------------------------
      |       |       | ||       |       |       | ||       |       |       |
4   6 |   5   |   7   | ||   9   |   1   | 4   6 | ||   3   |   8   |   2   |
      |       |       | ||       |       |       | ||       |       |       |
    3 |     3 |       | ||   2 3 |       |   2 3 | ||       |       |       |
    6 |       |   1   | ||     6 |   8   |     6 | ||   7   |   5   |   4   |
    9 |     9 |       | ||       |       |       | ||       |       |       |
    3 |     3 |       | ||     3 |       |       | ||       |       |       |
4     |       |   2   | || 4     |   5   |   7   | ||   6   |   1   |   9   |
  8   |   8   |       | ||       |       |       | ||       |       |       |
提示： [Locked Candidate] 数字 2 在宫 1-1 只位于行 3，故该行宫外位置不可为 2。
请输入命令 [n=执行提示, r=重新输入, e=退出]:
```
---

## Roadmap

- Add GUI support
- Export solutions

