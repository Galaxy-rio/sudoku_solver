#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple

Grid = List[List[int]]

def read_grid_from_stdin() -> Grid:
    grid: Grid = []
    print("请输入数独（9行，每行9个数字，0表示空）：")
    for r in range(9):
        line = input().strip()
        if len(line) != 9 or any(ch not in "0123456789" for ch in line):
            raise ValueError(f"第 {r+1} 行格式错误：应为9个字符，仅包含0-9。")
        grid.append([int(ch) for ch in line])
    return grid

def print_grid(grid: Grid) -> None:
    for r in range(9):
        row = ""
        for c in range(9):
            row += str(grid[r][c]) + (" " if c % 3 != 2 else " | ")
        print(row[:-3] if r % 3 == 2 else row)
        if r % 3 == 2 and r != 8:
            print("-" * 21)

def box_index(r: int, c: int) -> int:
    return (r // 3) * 3 + (c // 3)

def validate_initial(grid: Grid) -> None:
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if v == 0:
                continue
            bit = 1 << v
            b = box_index(r, c)
            if (rows[r] & bit) or (cols[c] & bit) or (boxes[b] & bit):
                raise ValueError(f"初始盘面冲突：在第{r+1}行第{c+1}列出现重复数字{v}。")
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

def solve_sudoku_all(grid: Grid):
    """回溯 + MRV + 位运算，找到所有解"""
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    empties: List[Tuple[int, int]] = []

    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if v == 0:
                empties.append((r, c))
            else:
                bit = 1 << v
                b = box_index(r, c)
                rows[r] |= bit
                cols[c] |= bit
                boxes[b] |= bit

    solutions = []

    def dfs():
        if not empties:
            # 保存当前解
            solutions.append([row[:] for row in grid])
            return

        best_i = -1
        best_mask = 0
        best_count = 10
        for i, (r, c) in enumerate(empties):
            used = rows[r] | cols[c] | boxes[box_index(r, c)]
            cand_mask = (~used) & 0b1111111110
            cnt = cand_mask.bit_count()
            if cnt == 0:
                return
            if cnt < best_count:
                best_count = cnt
                best_mask = cand_mask
                best_i = i
                if cnt == 1:
                    break

        r, c = empties.pop(best_i)
        b = box_index(r, c)

        mask = best_mask
        while mask:
            v_bit = mask & -mask
            mask ^= v_bit
            v = (v_bit.bit_length() - 1)

            grid[r][c] = v
            rows[r] |= v_bit
            cols[c] |= v_bit
            boxes[b] |= v_bit

            dfs()

            grid[r][c] = 0
            rows[r] ^= v_bit
            cols[c] ^= v_bit
            boxes[b] ^= v_bit

        empties.insert(best_i, (r, c))

    dfs()
    return solutions

def main():
    try:
        grid = read_grid_from_stdin()
        validate_initial(grid)
    except Exception as e:
        print("输入错误：", e)
        return

    solutions = solve_sudoku_all(grid)
    if solutions:
        print(f"\n总共找到 {len(solutions)} 个解：")
        for idx, sol in enumerate(solutions, 1):
            print(f"\n解 {idx}:")
            print_grid(sol)
    else:
        print("该数独无解。")

if __name__ == "__main__":
    main()
