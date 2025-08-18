#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict, deque
from typing import List, Set, Tuple, Dict, Optional

Grid = List[List[int]]
Pos = Tuple[int, int]

# ------------------------------
# 读入与基础设施
# ------------------------------

def read_board() -> Grid:
    """
    读取 9 行、每行 9 个字符（0-9），无分隔符；0 表示空格。
    """
    print("请输入数独（9行，每行9个数字，0表示空）：")
    board: Grid = []
    for r in range(9):
        line = input().strip()
        if len(line) != 9 or any(ch not in "0123456789" for ch in line):
            raise ValueError(f"第 {r+1} 行格式错误：应为9个字符，仅包含0-9。")
        board.append([int(ch) for ch in line])
    return board

def peers_and_units():
    """预计算每个格子的同伴（同行/列/宫）以及所有单位（27个unit）"""
    units = []  # 每个unit是一组坐标
    # 行
    for r in range(9):
        units.append([(r, c) for c in range(9)])
    # 列
    for c in range(9):
        units.append([(r, c) for r in range(9)])
    # 宫
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            units.append([(br + i, bc + j) for i in range(3) for j in range(3)])

    unit_of_cell: List[List[List[int]]] = [[[] for _ in range(9)] for _ in range(9)]
    for uid, unit in enumerate(units):
        for (r, c) in unit:
            unit_of_cell[r][c].append(uid)

    peers: List[List[Set[Pos]]] = [[set() for _ in range(9)] for _ in range(9)]
    for r in range(9):
        for c in range(9):
            ps = set()
            for uid in unit_of_cell[r][c]:
                for rc in units[uid]:
                    if rc != (r, c):
                        ps.add(rc)
            peers[r][c] = ps

    return units, unit_of_cell, peers

UNITS, UNIT_OF_CELL, PEERS = peers_and_units()

def rc_label(r: int, c: int) -> str:
    return f"R{r+1}C{c+1}"

# ------------------------------
# 候选生成与打印
# ------------------------------

def compute_base_candidates(board: Grid) -> List[List[Set[int]]]:
    """
    从当前盘面生成基础候选（不含手动消除）。
    已填格的候选为 {v}。
    """
    cand = [[set(range(1, 10)) for _ in range(9)] for _ in range(9)]
    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if v != 0:
                cand[r][c] = {v}
            else:
                used = set()
                # 行
                used |= set(board[r][i] for i in range(9))
                # 列
                used |= set(board[i][c] for i in range(9))
                # 宫
                br, bc = (r // 3) * 3, (c // 3) * 3
                used |= set(board[br + i][bc + j] for i in range(3) for j in range(3))
                used.discard(0)
                cand[r][c] -= used
    return cand

def apply_forbidden(cand: List[List[Set[int]]],
                    forbidden: List[List[Set[int]]]) -> None:
    """将手动消除过的候选（forbidden）应用到当前候选上。"""
    for r in range(9):
        for c in range(9):
            if forbidden[r][c]:
                cand[r][c] -= forbidden[r][c]

def pretty_print_candidates(board: Grid, cand: List[List[Set[int]]]) -> None:
    """
    以 3x3 小九宫形式打印每个格子的候选。
    已填格只在中心位置显示数字，其它位置留空。
    整体以 3x3 宫为块加粗分隔。
    """
    lines = []
    cell_sep = " | "
    block_sep_line = None

    for r in range(9):
        sublines = [""] * 3
        for c in range(9):
            box = [[" " for _ in range(3)] for _ in range(3)]
            if board[r][c] != 0:
                # 已填数字放中心
                box[1][1] = str(board[r][c])
            else:
                for n in cand[r][c]:
                    rr, cc = divmod(n - 1, 3)
                    box[rr][cc] = str(n)

            for k in range(3):
                sublines[k] += " ".join(box[k]) + cell_sep

            # 竖向粗分隔
            if c % 3 == 2 and c != 8:
                for k in range(3):
                    sublines[k] += "|| "

        lines.extend(sublines)

        # 横向粗分隔
        if r % 3 == 2 and r != 8:
            if block_sep_line is None:
                block_sep_line = "-" * len(sublines[0])
            lines.append(block_sep_line)

    print("\n".join(lines))

# ------------------------------
# 提示类型 1：Naked Single
# ------------------------------

def find_naked_single(board: Grid, cand: List[List[Set[int]]]) -> Optional[Tuple[str, Pos, int]]:
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0 and len(cand[r][c]) == 1:
                val = next(iter(cand[r][c]))
                text = f"[Naked Single] {rc_label(r,c)} 只能是 {val}。"
                return text, (r, c), val
    return None

# ------------------------------
# 提示类型 2：Hidden Single
# ------------------------------

def find_hidden_single(board: Grid, cand: List[List[Set[int]]]) -> Optional[Tuple[str, Pos, int]]:
    # 行 / 列 / 宫
    for uid, unit in enumerate(UNITS):
        count: Dict[int, List[Pos]] = {n: [] for n in range(1, 10)}
        for (r, c) in unit:
            if board[r][c] == 0:
                for n in cand[r][c]:
                    count[n].append((r, c))
        for n, locs in count.items():
            if len(locs) == 1:
                r, c = locs[0]
                kind = "行" if uid < 9 else ("列" if uid < 18 else "宫")
                scope = uid if uid < 9 else (uid - 9 if uid < 18 else uid - 18)
                if kind == "宫":
                    block_label = f"{(scope//3)+1}-{(scope%3)+1}"
                    text = f"[Hidden Single] 数字 {n} 在第{kind}{block_label} 内仅能放在 {rc_label(r,c)} → 填入 {n}。"
                elif kind == "行":
                    text = f"[Hidden Single] 数字 {n} 在第{kind}{scope+1} 内仅能放在 {rc_label(r,c)} → 填入 {n}。"
                else:
                    text = f"[Hidden Single] 数字 {n} 在第{kind}{scope+1} 内仅能放在 {rc_label(r,c)} → 填入 {n}。"
                return text, (r, c), n
    return None

# ------------------------------
# 提示类型 3：Simple Coloring（强链着色消除）
# ------------------------------

def build_strong_link_graph_for_digit(cand: List[List[Set[int]]], d: int) -> Dict[Pos, Set[Pos]]:
    """
    针对某个数字 d，构建强链图：
    在任一 行/列/宫 中，若 d 只出现于 2 个格子，则这两个格子间存在一条强链边。
    """
    G: Dict[Pos, Set[Pos]] = defaultdict(set)

    # 行
    for r in range(9):
        locs = [(r, c) for c in range(9) if d in cand[r][c]]
        if len(locs) == 2:
            a, b = locs
            G[a].add(b); G[b].add(a)

    # 列
    for c in range(9):
        locs = [(r, c) for r in range(9) if d in cand[r][c]]
        if len(locs) == 2:
            a, b = locs
            G[a].add(b); G[b].add(a)

    # 宫
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            locs = [(br+i, bc+j) for i in range(3) for j in range(3) if d in cand[br+i][bc+j]]
            if len(locs) == 2:
                a, b = locs
                G[a].add(b); G[b].add(a)

    return G

def two_color_components(G: Dict[Pos, Set[Pos]]) -> Dict[Pos, int]:
    """
    二色着色：同一连通分量用颜色 0/1。
    返回每个节点的颜色；无法二分（奇环）时可忽略（实战中一般也可二分）。
    """
    color: Dict[Pos, int] = {}
    for start in G.keys():
        if start in color:
            continue
        color[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in G[u]:
                if v not in color:
                    color[v] = 1 - color[u]
                    q.append(v)
                # 若出现冲突（同色相邻），这里不特殊处理
    return color

def find_simple_coloring_elimination(board: Grid,
                                     cand: List[List[Set[int]]]) -> Optional[Tuple[str, Pos, int, List[Pos]]]:
    """
    返回一条可执行的“消除候选”的提示：
    (描述文本, 目标格子, 被消除的数字d, 一条链路径（强链边序列上的节点列表，便于解释）)
    """
    # 对每个数字尝试
    for d in range(1, 10):
        # 构建强链图（仅由 d 的候选组成）
        G = build_strong_link_graph_for_digit(cand, d)
        if not G:
            continue
        color = two_color_components(G)
        if not color:
            continue

        # 收集每个连通分量节点
        comp_members: Dict[Pos, int] = {}     # node -> comp_id
        comp_nodes: List[List[Pos]] = []
        visited = set()
        for node in G.keys():
            if node in visited:
                continue
            # BFS 获取分量
            comp_id = len(comp_nodes)
            comp_nodes.append([])
            q = deque([node])
            visited.add(node)
            while q:
                u = q.popleft()
                comp_nodes[comp_id].append(u)
                comp_members[u] = comp_id
                for v in G[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)

        # 对每个分量，找“看到两种颜色”的外部格子进行消除
        for comp in comp_nodes:
            # 该分量中两种颜色的节点集合
            color_groups = {0: set(), 1: set()}
            for u in comp:
                color_groups[color[u]].add(u)

            # 所有含 d 的格子（包括分量外部）
            all_d_cells = [(r, c) for r in range(9) for c in range(9)
                           if board[r][c] == 0 and d in cand[r][c]]

            # 尝试找到一个目标格子 t：它同时是颜色0集与颜色1集的“可见格”的交集
            for t in all_d_cells:
                if t in comp:
                    continue  # 本分量中的节点不作为外部消除目标（也可做，但解释更复杂）
                sees0 = any(u in PEERS[t[0]][t[1]] for u in color_groups[0])
                sees1 = any(u in PEERS[t[0]][t[1]] for u in color_groups[1])
                if sees0 and sees1:
                    # 找一条链用于解释：从一个 0 色节点到一个 1 色节点的路径
                    path = find_any_path_across_colors(G, color_groups[0], color_groups[1])
                    # 生成说明文字
                    text = (
                        f"[Simple Coloring 强链着色] 针对数字 {d}：在若干行/列/宫中，{d} 只出现于两个位置构成强链。"
                        f"将该强链所在分量二色着色（A/B）。由于 {rc_label(*t)} 同时能看到两种颜色的 {d}，"
                        f"因此可在 {rc_label(*t)} 中**消除 {d}**。"
                    )
                    return text, t, d, path
    return None

def find_any_path_across_colors(G: Dict[Pos, Set[Pos]], groupA: Set[Pos], groupB: Set[Pos]) -> List[Pos]:
    """
    在强链图中找一条从 A 组任一点到 B 组任一点的路径，用于解释“链”。
    如果找不到，返回空列表。
    """
    if not groupA or not groupB:
        return []
    starts = list(groupA)
    goals = set(groupB)

    for s in starts:
        prev: Dict[Pos, Optional[Pos]] = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            if u in goals:
                # 回溯路径
                path = []
                cur = u
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path
            for v in G[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)
    return []

# ------------------------------
# 应用操作
# ------------------------------

def apply_fill(board: Grid, r: int, c: int, v: int) -> None:
    board[r][c] = v

def apply_eliminate(forbidden: List[List[Set[int]]], r: int, c: int, d: int) -> None:
    forbidden[r][c].add(d)

# ------------------------------
# 主交互循环
# ------------------------------

def main():
    # 初始读取
    try:
        board = read_board()
    except Exception as e:
        print("输入错误：", e)
        return

    # 记录“基于链/逻辑的手动消除”
    forbidden = [[set() for _ in range(9)] for _ in range(9)]

    while True:
        # 计算候选并应用手动消除
        cand = compute_base_candidates(board)
        apply_forbidden(cand, forbidden)

        # 打印当前候选棋盘
        print("\n当前候选棋盘（每格 3×3，中心为已填数字）：\n")
        pretty_print_candidates(board, cand)

        # 依次寻找下一条“人类逻辑提示”
        hint = (
            find_naked_single(board, cand) or
            find_hidden_single(board, cand) or
            None
        )

        chain_hint = None
        if hint is None:
            chain_hint = find_simple_coloring_elimination(board, cand)

        if hint is not None:
            text, (r, c), v = hint
            print("\n提示：", text)
            action = ("fill", (r, c, v))
        elif chain_hint is not None:
            text, target, d, path = chain_hint
            print("\n提示：", text)
            if path:
                chain_str = " — ".join(rc_label(*p) for p in path)
                print(f"链（强链边序列示意）：{chain_str}")
            print(f"建议操作：从 {rc_label(*target)} 的候选中**消除 {d}**")
            action = ("elim", (target[0], target[1], d))
        else:
            print("\n提示：未找到进一步的人类推理（需要更高级技巧或题目已近终局）。")
            action = None

        # 交互命令
        cmd = input("\n请输入命令 [n=执行提示, r=重新输入, e=退出]: ").strip().lower()
        if cmd == "n":
            if action is None:
                print("（当前没有可执行的提示。）")
            else:
                if action[0] == "fill":
                    r, c, v = action[1]
                    apply_fill(board, r, c, v)
                    # 填入后，该格不再需要“手动消除”记录
                    forbidden[r][c].clear()
                    print(f"已填入：{rc_label(r,c)} = {v}")
                else:
                    r, c, d = action[1]
                    apply_eliminate(forbidden, r, c, d)
                    print(f"已消除：{rc_label(r,c)} 中的候选 {d}")
        elif cmd == "r":
            try:
                board = read_board()
                forbidden = [[set() for _ in range(9)] for _ in range(9)]
            except Exception as e:
                print("输入错误：", e)
                return
        elif cmd == "e":
            print("已退出。")
            return
        else:
            print("无效输入，请输入 n / r / e。")

if __name__ == "__main__":
    main()
